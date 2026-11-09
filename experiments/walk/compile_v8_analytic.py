"""Compile walking code v8 into a sparse, phase-gated tanh controller.

Pass the nominal code-v8 frozen replay JSON and an output JSON path. This is
a source-specific compiler, not a general Python-to-weights translator.
The first layer reconstructs the teacher's phase on the existing 20 ms grid
for a 5 s rollout and approximates its elementary scalar functions. The
second layer gates the stance and swing equations; only the local linear
approximation of stance ankle geometry uses states from the supplied replay.
The exported spec runs unchanged through walk_policy.compile_weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from walk_policy import FEATURES, compile_weights


class Expr:
    """Sparse affine expression over the previous layer."""

    def __init__(self, terms=None, bias=0.0):
        self.terms = {} if terms is None else terms
        self.bias = bias

    def __mul__(self, c):
        return Expr({i: v * c for i, v in self.terms.items()}, self.bias * c)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __add__(self, other):
        if not isinstance(other, Expr):
            other = Expr(bias=float(other))
        terms = self.terms.copy()
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0.0) + v
        return Expr(terms, self.bias + other.bias)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def row(self):
        return {"terms": [{"feature": i, "coef": float(v)} for i, v in sorted(self.terms.items()) if v],
                "bias": float(self.bias)}


def layer():
    rows = []

    def add(expr):
        result = Expr({len(rows): 1.0})
        rows.append(expr.row())
        return result

    return rows, add


def project(A, y):
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.mean((A @ w - y)**2), w


def scalar_fit(x, y, basis, grid):
    best = (float("inf"), None, None)
    for parameter in grid:
        score, w = project(basis(x, parameter), y)
        if score < best[0]:
            best = (score, parameter, w)
    return best[1], best[2]


def fit_hump():
    """Fit sin(pi*u) with two symmetric pairs of tanh features."""
    x = np.linspace(0, 1, 201) - .5
    y = np.cos(np.pi*x)
    atoms = [(k, c) for k in np.linspace(1.5, 9, 22) for c in np.linspace(.04,.49,22)]
    A = np.array([np.tanh(k*(x+c))-np.tanh(k*(x-c)) for k,c in atoms]).T
    # Keep the best-conditioned pair: highly cancelling solutions lose stability.
    G = A.T @ A
    R = A.T @ y
    diag = G.diagonal()
    a = diag[:,None]
    b = diag[None,:]
    den = a*b-G*G
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = (b*R[:,None]-G*R[None,:])/den
        w2 = (a*R[None,:]-G*R[:,None])/den
        loss = np.dot(y,y)-w1*R[:,None]-w2*R[None,:]
    loss[(den<1e-9) | (abs(w1)>16) | (abs(w2)>16)] = np.inf
    i,j = np.unravel_index(np.argmin(loss),loss.shape)
    return [(atoms[i],w1[i,j]),(atoms[j],w2[i,j])]


def fit_startup():
    """Fit first-cycle knee ramp correction using five time-only tanhs.

    The target is the difference between the right knee's half-wave with and
    without the teacher's startup ramp.
    """
    t = np.arange(251)*.02
    y = np.maximum(0,np.sin(7*t)) * (np.minimum(1,2*t)-1)
    # tanh(k(t-c)) - 1 vanishes after startup; stage first samples densely.
    centers = np.arange(.01,.45,.02)
    slopes = (6,9,13,18,26,38,55)
    atoms = [(k,c) for k in slopes for c in centers]
    A = np.array([np.tanh(k*(t-c))-1 for k,c in atoms]).T
    selected = []
    residual = y.copy()
    for _ in range(5):
        score = abs(A.T@residual)/np.maximum(1e-12,np.linalg.norm(A,axis=0))
        if selected:
            score[selected] = -1
        selected.append(int(np.argmax(score)))
        _, w = project(A[:,selected], y)
        residual = y-A[:,selected]@w
    return [(atoms[i],v) for i,v in zip(selected,w)]


def compile_analytic(trace, *, mask=14.0, eps=.002):
    L1, one = layer()
    L2, two = layer()
    x = {k:Expr({i:1.0}) for i,k in enumerate(FEATURES)}
    t = x["t"]

    def copy(expr):
        return (1 / eps) * one(eps * expr)

    # The teacher's branch schedule is sin(7*t)>0.  In the 20 ms simulator
    # time grid, place each boundary between adjacent decision instants.
    T = math.pi/7
    cut = [.01] + [.02*(math.floor(i*T/.02)+.5) for i in range(1,12)]
    H = [(one(2200*(t-c)) + 1)*.5 for c in cut]
    n = sum(H[1:])
    p_left = 1 - H[0] + sum(((-1)**(i+1))*H[i] for i in range(1,len(H)))
    u = copy(t) * (1/T) - n

    times = np.arange(251)*.02
    def ramp_basis(tt, param):
        k,c = param
        return np.c_[np.ones(len(tt)),np.tanh(k*(tt-c)),tt>.001,tt>cut[1],tt>cut[2]]

    (k,c), ramp_w = scalar_fit(times, np.minimum(1,2*times),ramp_basis,
                             [(k,c) for k in np.linspace(3.8,8.5,25) for c in np.linspace(.18,.30,25)])
    a = ramp_w[0]+ramp_w[1]*one(k*(t-c)) + ramp_w[2]*H[0] + ramp_w[3]*H[1] + ramp_w[4]*H[2]

    pitch = np.linspace(-.50,.15,151)
    def min_basis(pp,param):
        k,c = param
        return np.c_[np.ones(len(pp)),pp,np.tanh(k*(pp-c))]

    (k,c), mw = scalar_fit(pitch,.7*np.minimum(0,pitch+.3),min_basis,
                          [(k,c) for k in np.linspace(5,25,30) for c in np.linspace(-.43,-.2,30)])
    m = mw[0]+mw[2]*one(k*(x["pitch"]-c))
    b = copy(-.7*x["pitch"]-.025*x["pitch_v"]) - .245*a
    hip = [copy(-.7*x[f"q{i}"]-.018*x[f"dq{i}"]+mw[1]*x["pitch"])+m for i in (0,3)]
    knee = [copy(-.8*x[f"q{i+1}"]-.022*x[f"dq{i+1}"]-.064) for i in (0,3)]

    stance = []
    swing = []
    for i in (0,3):
        # Only nominal teacher states in the fit, used for local linear
        # approximation of the trigonometric stance-ankle geometry term.
        samples = np.array([[r["q"][i]-r["pitch"],r["q"][i+1],r["vx"],r["dq"][i+2]]
                            for r in trace if (math.sin(7*r["t"])>0) == (i==3)])
        g = .4*np.sin(samples[:,0])+.4*np.sin(samples[:,0]+samples[:,1])
        _, w = project(np.c_[samples[:,:2],np.ones(len(samples))],g)
        stance.append(copy(w[0]*(x[f"q{i}"]-x["pitch"])+w[1]*x[f"q{i+1}"]+w[2]
                          -.3*x["vx"]+.1-.01*x[f"dq{i+2}"]))
        swing.append(copy(.3*(x["pitch"]-x[f"q{i}"]-x[f"q{i+1}"]-x[f"q{i+2}"])
                          -.01*x[f"dq{i+2}"]))

    assert len(L1) == 24
    # tanh near zero is a linear relay. A large offset saturates the
    # inactive branch; subtract the matching baseline before amplification.
    K = [mask*p_left, mask*(1-p_left)]
    offsets = [two(k) for k in K]

    def gated(expr,i):
        return (two(eps*expr+K[i])-offsets[i])*(1/eps)

    hipmid = two(eps*b)*(1/eps)
    hipdelta = gated(hip[0]-b,0)-gated(hip[1]-b,1)
    ankle = [gated(swing[i],i)+gated(stance[i],1-i) for i in range(2)]
    hump = fit_hump()
    knee_out = []
    for i in range(2):
        wave = sum(v*(two(k*(u-.5+c)+K[i])-two(k*(u-.5-c)+K[i]))
                   for (k,c),v in hump)
        if i == 0:
            # u+n reconstructs normalized time even across phase changes.
            time = T*(u+n)
            wave += sum(v*(two(k*(time-c))-1) for (k,c),v in fit_startup())
        knee_out.append(two(eps*knee[i])*(1/eps)-.72*wave)

    outputs = [hipmid+hipdelta,knee_out[0],ankle[0],hipmid-hipdelta,knee_out[1],ankle[1]]
    assert len(L2) == 24
    return {"layers":[{"rows":L1},{"rows":L2},{"rows":[y.row() for y in outputs]}]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("teacher", type=Path, help="nominal code-v8 frozen replay JSON")
    p.add_argument("output", type=Path)
    args = p.parse_args()
    trace = json.loads(args.teacher.read_text())["result"]["trace"]
    spec = compile_analytic(trace)
    compile_weights(spec)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(spec,indent=2)+"\n")


if __name__ == "__main__":
    main()
