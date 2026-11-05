"""
Session 14: the step hook that balances and watches the experts.

Plugs into train.loop.run_training(step_hook=...), which calls before(model, step) and
after(model, step) around every optimiser step.

    before   zero each MoE layer's token counter
    after    the counters now hold the load of the WHOLE step (all micro-batches): this is
             whole-batch balancing (notes Section 14).
             If gamma > 0, update each expert's bias:  b_i += gamma * sign(even - load_i)
             (notes Section 13). Then record, per layer:
                 load of every expert (share of the step's k*N routing slots; even = 1/E)
                 MaxVio = (max load - even) / even
                 starved = experts below starve_frac of the even share; zero = experts with no tokens
                 bias min / max, and the sequence balance loss
One .cpu() per step, so the cost is small.
"""
import csv

import torch

from model.moe import moe_layers


class MoEBalancer:
    def __init__(self, model, gamma=1e-3, starve_frac=0.1):
        self.mods = moe_layers(model)
        assert self.mods, "model has no MoE layers"
        self.E = self.mods[0].n_experts
        self.gamma = gamma
        self.starve_frac = starve_frac
        self.rows = []                 # (step, tensor [L, E + 6])

    @torch.no_grad()
    def before(self, model, step):
        for m in self.mods:
            m.counts.zero_()

    @torch.no_grad()
    def after(self, model, step):
        c = torch.stack([m.counts for m in self.mods])                     # [L, E]
        even = 1.0 / self.E
        load = c / c.sum(1, keepdim=True).clamp(min=1)
        if self.gamma:
            delta = self.gamma * torch.sign(even - load)
            for m, d in zip(self.mods, delta):
                m.bias += d
        bias = torch.stack([m.bias for m in self.mods])
        aux = torch.stack([m.last_aux.float() if m.last_aux is not None
                           else torch.zeros((), device=c.device) for m in self.mods])
        packed = torch.cat([
            load,
            ((load.max(1).values - even) / even)[:, None],
            (load < self.starve_frac * even).sum(1, keepdim=True).float(),
            (c == 0).sum(1, keepdim=True).float(),
            bias.min(1, keepdim=True).values,
            bias.max(1, keepdim=True).values,
            aux[:, None],
        ], 1).cpu()
        self.rows.append((step, packed))

    FIELDS = ["maxvio", "starved", "zero", "bias_min", "bias_max", "aux"]

    def save(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "layer"] + self.FIELDS + [f"load{e}" for e in range(self.E)])
            for step, p in self.rows:
                for layer, r in enumerate(p.tolist()):
                    E = self.E
                    w.writerow([step, layer] + [f"{v:.6g}" for v in r[E:]] + [f"{v:.6g}" for v in r[:E]])


def read_stats(path):
    """CSV from MoEBalancer.save -> dict of tensors: steps [S], each field [S, L], load [S, L, E]."""
    by_step = {}
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        E = sum(1 for k in rd.fieldnames if k.startswith("load"))
        for r in rd:
            by_step.setdefault(int(r["step"]), {})[int(r["layer"])] = r
    steps = sorted(by_step)
    L = len(by_step[steps[0]])
    out = {"steps": torch.tensor(steps)}
    for k in MoEBalancer.FIELDS:
        out[k] = torch.tensor([[float(by_step[s][l][k]) for l in range(L)] for s in steps])
    out["load"] = torch.tensor([[[float(by_step[s][l][f"load{e}"]) for e in range(E)]
                                 for l in range(L)] for s in steps])
    return out
