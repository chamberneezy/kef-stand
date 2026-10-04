#!/usr/bin/env python3
"""Parametric tilted desk/shelf stand for the KEF LSX II LT (Ender 5 Plus friendly).

Pure numpy/scipy/matplotlib - no CAD library needed. Run:  python3 make_stand.py
Writes kef_lsx_ii_lt_stand.stl next to this script and prints a report.

Axes (mm): X = width (left-right), Y = depth (0 = front, speaker faces -Y),
           Z = up. The part prints flat on the bed, sloped top facing up.

Method: a 2D Delaunay triangulation of the footprint is used as a height-field.
Every triangle gets a top height (sloped plane, lowered inside the pocket) and a
bottom height (flat, raised in the screw counterbore and foot recesses). Vertical
walls are emitted wherever neighbouring triangles differ, so the mesh is
watertight by construction.
"""
import math
import struct
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.spatial import Delaunay, cKDTree

# ------------------------------------------------------------------ parameters
P = dict(
    W=182.0,            # stand width  (speaker is 155)
    D=196.0,            # stand depth  (speaker is 180)
    R=14.0,             # outer corner radius (soft corners)
    fillet=4.0,         # rounded top edge, radius in mm
    tilt=8.0,           # upward tilt in degrees (KEF's own P1 pad uses 10)
    t_back=10.0,        # thickness at the thin (rear) edge
    pocket_depth=3.0,   # locating pocket the speaker sits in
    pk_w=158.0,         # pocket width  (speaker 155 + 3 clearance)
    pk_d=182.0,         # pocket depth  (speaker 180 + 2 clearance)
    pk_r=4.0,           # pocket corner radius
    slot_w=7.0,         # clearance slot for the 1/4"-20 screw
    slot_y0=25.0,       # slot start, measured from the front edge
    slot_y1=155.0,      # slot end
    cb_w=18.0,          # counterbore (screw head / washer) width, underside
    cb_depth=7.0,       # counterbore depth from the underside
    foot_d=22.0,        # round recess for self-adhesive rubber feet
    foot_depth=1.2,
    foot_inset_x=22.0,
    foot_inset_y=22.0,
    step=0.5,           # outline sampling distance
    fill=3.0,           # interior point spacing
)


def zs(y, p=P):
    """Top (sloped) surface height at depth y."""
    return p["t_back"] + (p["D"] - y) * math.tan(math.radians(p["tilt"]))


def edge_drop(x, y, p=P):
    """How far the top surface is rounded down at (x, y): quarter-circle fillet along the outer edge."""
    r = p["fillet"]
    if r <= 0:
        return 0.0
    dx = abs(x - p["W"] / 2) - (p["W"] / 2 - p["R"])
    dy = abs(y - p["D"] / 2) - (p["D"] / 2 - p["R"])
    sdf = math.hypot(max(dx, 0.0), max(dy, 0.0)) + min(max(dx, dy), 0.0) - p["R"]
    d = max(-sdf, 0.0)
    if d >= r:
        return 0.0
    return r - math.sqrt(max(r * r - (r - d) ** 2, 0.0))


# ------------------------------------------------------------------ 2D shapes
def _resample(poly, step):
    pts = []
    n = len(poly)
    for i in range(n):
        a, b = np.array(poly[i]), np.array(poly[(i + 1) % n])
        k = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        for t in range(k):
            pts.append(a + (b - a) * t / k)
    return np.array(pts)


def rrect(cx, cy, w, h, r, step):
    r = min(r, w / 2, h / 2)
    corners = [(cx + w / 2 - r, cy + h / 2 - r, 0), (cx - w / 2 + r, cy + h / 2 - r, 90),
               (cx - w / 2 + r, cy - h / 2 + r, 180), (cx + w / 2 - r, cy - h / 2 + r, 270)]
    poly = []
    na = max(4, int(math.ceil((math.pi / 2 * r) / step)))
    for ccx, ccy, a0 in corners:
        for k in range(na + 1):
            a = math.radians(a0 + 90 * k / na)
            poly.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    return _resample(poly, step)


def circle(cx, cy, d, step):
    n = max(24, int(math.ceil(math.pi * d / step)))
    return np.array([(cx + d / 2 * math.cos(2 * math.pi * k / n),
                      cy + d / 2 * math.sin(2 * math.pi * k / n)) for k in range(n)])


def stadium(cx, y0, y1, w, step):
    """Slot with round ends spanning y0..y1 (full extent) and width w, along Y."""
    r = w / 2
    c0, c1 = y0 + r, y1 - r
    poly = []
    na = max(8, int(math.ceil(math.pi * r / step)))
    for k in range(na + 1):  # back cap
        a = math.radians(0 + 180 * k / na)
        poly.append((cx + r * math.cos(a), c1 + r * math.sin(a)))
    for k in range(na + 1):  # front cap
        a = math.radians(180 + 180 * k / na)
        poly.append((cx + r * math.cos(a), c0 + r * math.sin(a)))
    return _resample(poly, step)


# ------------------------------------------------------------------ build mesh
def build(p=P):
    W, D, st = p["W"], p["D"], p["step"]
    foot_poly = rrect(W / 2, D / 2, W, D, p["R"], st)
    pocket_poly = rrect(W / 2, D / 2, p["pk_w"], p["pk_d"], p["pk_r"], st)
    slot_poly = stadium(W / 2, p["slot_y0"], p["slot_y1"], p["slot_w"], st)
    cb_poly = stadium(W / 2, p["slot_y0"] - (p["cb_w"] - p["slot_w"]) / 2,
                      p["slot_y1"] + (p["cb_w"] - p["slot_w"]) / 2, p["cb_w"], st)
    fx, fy = p["foot_inset_x"], p["foot_inset_y"]
    feet_polys = [circle(x, y, p["foot_d"], st)
                  for x in (fx, W - fx) for y in (fy, D - fy)]

    rings = [rrect(W / 2, D / 2, W - 2 * d, D - 2 * d, max(p["R"] - d, 1.0), 0.7)
             for d in (p["fillet"] * f for f in (0.07, 0.17, 0.3, 0.45, 0.62, 0.8, 1.0))] if p["fillet"] > 0 else []
    outlines = [foot_poly, pocket_poly, slot_poly, cb_poly] + feet_polys + rings
    pts = np.vstack(outlines)

    # interior fill points, kept away from outlines
    tree = cKDTree(pts)
    gx, gy = np.meshgrid(np.arange(p["fill"] / 2, W, p["fill"]), np.arange(p["fill"] / 2, D, p["fill"]))
    g = np.c_[gx.ravel(), gy.ravel()]
    ok = MPath(foot_poly).contains_points(g) & ~MPath(slot_poly).contains_points(g)
    g = g[ok]
    d, _ = tree.query(g)
    g = g[d > 1.2]
    pts = np.vstack([pts, g])
    pts = np.unique(np.round(pts, 4), axis=0)

    tri = Delaunay(pts).simplices
    cen = pts[tri].mean(axis=1)
    keep = MPath(foot_poly).contains_points(cen) & ~MPath(slot_poly).contains_points(cen)
    tri = tri[keep]
    cen = cen[keep]
    # make CCW
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]

    in_pocket = MPath(pocket_poly).contains_points(cen)
    in_cb = MPath(cb_poly).contains_points(cen)
    in_foot = np.zeros(len(cen), bool)
    for fp in feet_polys:
        in_foot |= MPath(fp).contains_points(cen)
    bot = np.where(in_cb, p["cb_depth"], np.where(in_foot, p["foot_depth"], 0.0))
    pk = in_pocket.astype(float)

    def top_z(i, t):  # vertex index i, triangle t
        return zs(pts[i][1], p) - p["pocket_depth"] * pk[t] - edge_drop(pts[i][0], pts[i][1], p)

    tris = []

    def quad(q0, q1, q2, q3, hint):
        n = np.cross(q1 - q0, q2 - q0)
        if np.dot(n, hint) < 0:
            q0, q1, q2, q3 = q3, q2, q1, q0
        tris.append((q0, q1, q2))
        tris.append((q0, q2, q3))

    edges = {}
    for t, (i, j, k) in enumerate(tri):
        pt = lambda v, z: np.array([pts[v][0], pts[v][1], z])
        tris.append((pt(i, top_z(i, t)), pt(j, top_z(j, t)), pt(k, top_z(k, t))))
        tris.append((pt(i, bot[t]), pt(k, bot[t]), pt(j, bot[t])))
        for u, v in ((i, j), (j, k), (k, i)):
            edges.setdefault((min(u, v), max(u, v)), []).append((t, u, v))

    for (_, _), lst in edges.items():
        t, u, v = lst[0]
        d = pts[v] - pts[u]
        out_a = np.array([d[1], -d[0], 0.0])  # outward normal of triangle t (CCW -> right side)
        U = lambda z: np.array([pts[u][0], pts[u][1], z])
        V = lambda z: np.array([pts[v][0], pts[v][1], z])
        if len(lst) == 1:
            quad(U(bot[t]), V(bot[t]), V(top_z(v, t)), U(top_z(u, t)), out_a)
        else:
            t2 = lst[1][0]
            # top step
            ta_u, ta_v = top_z(u, t), top_z(v, t)
            tb_u, tb_v = top_z(u, t2), top_z(v, t2)
            if abs(ta_u - tb_u) > 1e-9 or abs(ta_v - tb_v) > 1e-9:
                if ta_u > tb_u:
                    quad(U(tb_u), V(tb_v), V(ta_v), U(ta_u), out_a)
                else:
                    quad(U(ta_u), V(ta_v), V(tb_v), U(tb_u), -out_a)
            # bottom step
            if abs(bot[t] - bot[t2]) > 1e-9:
                if bot[t] < bot[t2]:
                    quad(U(bot[t]), V(bot[t]), V(bot[t2]), U(bot[t2]), out_a)
                else:
                    quad(U(bot[t2]), V(bot[t2]), V(bot[t]), U(bot[t]), -out_a)
    return np.array(tris, dtype=np.float64)


# ------------------------------------------------------------------ output + checks
def write_stl(path, tris, name=b"KEF LSX II LT stand"):
    n = len(tris)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ln[ln == 0] = 1
    nrm = nrm / ln[:, None]
    dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    arr = np.zeros(n, dt)
    arr["n"] = nrm
    arr["v"] = tris
    with open(path, "wb") as f:
        f.write(name.ljust(80, b" ")[:80])
        f.write(struct.pack("<I", n))
        f.write(arr.tobytes())


def check(tris, p=P):
    keys = np.round(tris * 1e4).astype(np.int64)
    uniq, inv = np.unique(keys.reshape(-1, 3), axis=0, return_inverse=True)
    idx = inv.reshape(-1, 3)
    # drop degenerate triangles
    good = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    idx = idx[good]
    directed = {}
    for a, b, c in idx:
        for u, v in ((a, b), (b, c), (c, a)):
            directed[(u, v)] = directed.get((u, v), 0) + 1
    bad = 0
    for (u, v), n in directed.items():
        if n != 1 or directed.get((v, u), 0) != 1:
            bad += 1
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return dict(triangles=len(tris), bad_edges=bad, volume_cm3=vol / 1000.0,
                bbox_min=tris.reshape(-1, 3).min(0).round(2).tolist(),
                bbox_max=tris.reshape(-1, 3).max(0).round(2).tolist())


if __name__ == "__main__":
    out = Path(__file__).with_name("kef_lsx_ii_lt_stand.stl")
    tris = build()
    write_stl(out, tris)
    rep = check(tris)
    print(out, out.stat().st_size, "bytes")
    for k, v in rep.items():
        print(f"{k}: {v}")
    print("front height (mm):", round(zs(0) , 2), " rear height (mm):", round(zs(P['D']), 2))
    sys.exit(0 if rep["bad_edges"] == 0 and rep["volume_cm3"] > 0 else 1)
