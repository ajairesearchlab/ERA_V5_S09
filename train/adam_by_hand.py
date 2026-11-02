"""
Adam computed by hand, in plain Python floats (fp64), plus small helpers used by
Session 11. No torch in this file on purpose: every number here can be checked
with a calculator, and PyTorch is then checked against it.

    hand_adam            m, v, m_hat, v_hat, step and w for every step (Section 6 of the notes)
    bias_factor          how much bigger the UNcorrected step is than the corrected one
    steps_until_small    after how many steps that factor stays within tol of 1
    parabola_vertex      minimum of a parabola through 3 equally spaced points (lr sweep)
    loglog_fit           straight line through log(x), log(y)  (lr* against width)
    first_sustained      first index from which a condition holds to the end
    rolling_median       simple smoother for noisy per-step logs

Self-test (reproduces the five-step table in the notes):
    python -m train.adam_by_hand
"""
import math


def hand_adam(grads, w0, lr, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, bias_correction=True):
    """One weight, one gradient per step. Same order of operations as torch.optim.AdamW:
    decoupled decay uses the weight from before the step, then the Adam step."""
    m = v = 0.0
    w = w0
    rows = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g                 # first moment: average gradient
        v = b2 * v + (1 - b2) * g * g             # second moment: average squared gradient
        if bias_correction:
            m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        else:
            m_hat, v_hat = m, v
        before = w
        w = w - lr * wd * w                       # decoupled decay: nothing divides it
        w = w - lr * m_hat / (math.sqrt(v_hat) + eps)
        rows.append({"t": t, "g": g, "m": m, "v": v, "m_hat": m_hat, "v_hat": v_hat,
                     "step": w - before, "w": w})
    return rows


def bias_factor(t, b1, b2):
    """Uncorrected step / corrected step at step t, for the same m and v (eps ignored).
    m/sqrt(v) = m_hat (1 - b1^t) / (sqrt(v_hat) sqrt(1 - b2^t))."""
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


def steps_until_small(b1, b2, tol, horizon=200_000):
    """First step t after which |bias_factor - 1| < tol for every later step."""
    last_bad = 0
    for t in range(1, horizon + 1):
        if abs(bias_factor(t, b1, b2) - 1) >= tol:
            last_bad = t
    return last_bad + 1


def parabola_vertex(xs, ys):
    """xs equally spaced (e.g. log2 lr). Returns x of the minimum of the parabola through
    the three points, clamped to the outer two."""
    (x0, x1, x2), (y0, y1, y2) = xs, ys
    h = x1 - x0
    curv = y0 - 2 * y1 + y2
    if curv <= 0:                                  # not a valley: keep the middle point
        return x1
    x = x1 + h * (y0 - y2) / (2 * curv)
    return min(max(x, x0), x2)


def loglog_fit(xs, ys):
    """Least squares for log(y) = a + b log(x). Returns a, b, r2.
    r2 is NaN when every y is the same (a flat line explains nothing)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    sxx = sum((x - mx) ** 2 for x in lx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ly)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(lx, ly))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-24 else float("nan")
    return a, b, r2


def first_sustained(values, cond):
    """First index i such that cond(values[j]) is True for every j >= i; None if never."""
    idx = None
    for i in range(len(values) - 1, -1, -1):
        if cond(values[i]):
            idx = i
        else:
            break
    return idx


def rolling_median(xs, k=11):
    half = k // 2
    out = []
    for i in range(len(xs)):
        win = sorted(xs[max(0, i - half):i + half + 1])
        out.append(win[len(win) // 2])
    return out


if __name__ == "__main__":
    rows = hand_adam([0.50, 0.40, 0.60, 0.45, 0.55], w0=1.0, lr=1e-3)
    print(f"{'t':>2}{'g':>6}{'m':>9}{'v':>11}{'m_hat':>9}{'v_hat':>9}{'step':>12}{'w':>11}")
    for r in rows:
        print(f"{r['t']:>2}{r['g']:>6.2f}{r['m']:>9.4f}{r['v']:>11.6f}{r['m_hat']:>9.4f}"
              f"{r['v_hat']:>9.4f}{r['step']:>12.6f}{r['w']:>11.6f}")
    for b2 in (0.999, 0.95):
        print(f"beta2={b2}: factor t=1 {bias_factor(1, 0.9, b2):.2f}, t=20 {bias_factor(20, 0.9, b2):.2f}, "
              f"within 1% after {steps_until_small(0.9, b2, 0.01)} steps")
