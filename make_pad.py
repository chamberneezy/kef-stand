#!/usr/bin/env python3
"""Parametric tilted desk pad for the KEF LSX II LT, in the style of KEF's P1 pad.

A replica of the P1's shape: a thin rounded plate that flares up to a tilted round
platform with a shallow inset disc. The speaker sits on the platform and is held by
one 1/4"-20 screw from below, placed off-centre in the disc as on the P1. Run:
python3 make_pad.py
Writes kef_lsx_ii_lt_pad.stl next to this script and prints a report.

Axes and method are the same as make_stand.py (height-field over a 2D Delaunay
triangulation). Differences: the top is a smooth per-vertex surface whose only
step is the inset disc, and the counterbore ceiling is a sloped plane parallel to
the platform, so the screw head seats square to the tilted screw.
"""
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy.spatial import Delaunay, cKDTree

from make_stand import check, circle, edge_drop, rrect, stadium, write_stl

# ------------------------------------------------------------------ parameters
P = dict(
    W=146.7,            # pad width  (KEF P1 size; speaker is 154.5)
    D=167.5,            # pad depth  (KEF P1 size; speaker is 180)
    R=30.0,             # outer corner radius
    fillet=1.5,         # rounded top edge, radius in mm
    edge_t=3.0,         # plate thickness at the outer edge
    tilt=8.0,           # upward tilt in degrees (KEF's own P1 pad uses 10)
    disc_d=66.0,        # platform diameter (where the flare starts)
    disc_y=112.0,       # platform centre, measured from the front edge
    disc_h=20.0,        # platform height at its centre
    well_d=60.0,        # inset disc inside the platform (takes a felt or rubber disc)
    well_depth=1.0,
    flare=1.4,          # 1 = straight cone from platform to edge, higher = more concave
    screw_off=-19.0,    # screw position relative to the platform centre (negative = toward the front)
    slot_w=7.0,         # clearance hole for the 1/4"-20 screw
    seat_t=6.0,         # platform plane down to the screw head seat
    cb_w=20.0,          # counterbore (screw head / washer) width, underside
    foot_d=22.0,        # round recess for self-adhesive rubber feet
    foot_depth=1.2,
    foot_inset_x=26.0,
    foot_inset_y=26.0,
    step=0.5,           # outline sampling distance
    fill=2.0,           # interior point spacing
)


def zd(y, p=P):
    """Platform plane height at depth y. The speaker's underside lies in this plane."""
    return p["disc_h"] + (p["disc_y"] - y) * math.tan(math.radians(p["tilt"]))


def sdf(x, y, p=P):
    """Signed distance to the rounded-rectangle footprint (negative inside)."""
    dx = np.abs(x - p["W"] / 2) - (p["W"] / 2 - p["R"])
    dy = np.abs(y - p["D"] / 2) - (p["D"] / 2 - p["R"])
    return np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)) + np.minimum(np.maximum(dx, dy), 0.0) - p["R"]


def top_surface(x, y, p=P):
    """Top height: tilted platform, then a flare that falls to edge_t at the outline.

    The flare is parametrised along rays from the platform centre (0 at the platform
    rim, 1 at the outline), which keeps it smooth all the way round.
    """
    cx, cy, rd = p["W"] / 2, p["disc_y"], p["disc_d"] / 2
    r = np.hypot(x - cx, y - cy)
    rs = np.where(r > 0, r, 1.0)
    ux, uy = (x - cx) / rs, (y - cy) / rs
    lo, hi = np.zeros_like(r), np.full_like(r, p["W"] + p["D"])
    for _ in range(50):  # distance along the ray to the outline
        mid = (lo + hi) / 2
        inside = sdf(cx + mid * ux, cy + mid * uy, p) < 0
        lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
    t = np.clip((r - rd) / np.maximum(lo - rd, 1e-9), 0.0, 1.0)
    rim = zd(cy + rd * uy, p)
    z = np.where(r <= rd, zd(y, p), p["edge_t"] + (rim - p["edge_t"]) * (1 - t) ** p["flare"])
    return z - np.array([edge_drop(a, b, p) for a, b in zip(x, y)])


# ------------------------------------------------------------------ build mesh
def build(p=P):
    W, D, st = p["W"], p["D"], p["step"]
    cx, cy = W / 2, p["disc_y"]
    tan = math.tan(math.radians(p["tilt"]))
    foot_poly = rrect(W / 2, D / 2, W, D, p["R"], st)
    disc_poly = circle(cx, cy, p["disc_d"], st)
    well_poly = circle(cx, cy, p["well_d"], st)
    # the screw is square to the platform, so it leans forward by tilt on its way down
    sy = cy + p["screw_off"]
    slot_poly = stadium(cx, sy - p["seat_t"] * tan - p["slot_w"] / 2, sy + p["slot_w"] / 2, p["slot_w"], st)
    cb_poly = stadium(cx, sy - zd(sy, p) * tan - p["cb_w"] / 2, sy - p["seat_t"] * tan + p["cb_w"] / 2,
                      p["cb_w"], st)
    fx, fy = p["foot_inset_x"], p["foot_inset_y"]
    feet_polys = [circle(x, y, p["foot_d"], st)
                  for x in (fx, W - fx) for y in (fy, D - fy)]

    rings = [rrect(W / 2, D / 2, W - 2 * d, D - 2 * d, max(p["R"] - d, 1.0), 0.7)
             for d in (p["fillet"] * f for f in (0.07, 0.17, 0.3, 0.45, 0.62, 0.8, 1.0))] if p["fillet"] > 0 else []
    outlines = [foot_poly, disc_poly, well_poly, slot_poly, cb_poly] + feet_polys + rings
    pts = np.vstack(outlines)

    # interior fill points, kept away from outlines
    tree = cKDTree(pts)
    gx, gy = np.meshgrid(np.arange(p["fill"] / 2, W, p["fill"]), np.arange(p["fill"] / 2, D, p["fill"]))
    g = np.c_[gx.ravel(), gy.ravel()]
    ok = MPath(foot_poly).contains_points(g) & ~MPath(slot_poly).contains_points(g)
    g = g[ok]
    d, _ = tree.query(g)
    g = g[d > 0.9]
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

    in_well = MPath(well_poly).contains_points(cen)
    in_cb = MPath(cb_poly).contains_points(cen)
    in_foot = np.zeros(len(cen), bool)
    for fp in feet_polys:
        in_foot |= MPath(fp).contains_points(cen)

    top = top_surface(pts[:, 0], pts[:, 1], p)
    seat = np.array([zd(y, p) for y in pts[:, 1]]) - p["seat_t"]

    def top_z(i, t):  # vertex index i, triangle t
        return top[i] - p["well_depth"] * in_well[t]

    def bot_z(i, t):
        return seat[i] if in_cb[t] else (p["foot_depth"] if in_foot[t] else 0.0)

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
        tris.append((pt(i, bot_z(i, t)), pt(k, bot_z(k, t)), pt(j, bot_z(j, t))))
        for u, v in ((i, j), (j, k), (k, i)):
            edges.setdefault((min(u, v), max(u, v)), []).append((t, u, v))

    for (_, _), lst in edges.items():
        t, u, v = lst[0]
        d = pts[v] - pts[u]
        out_a = np.array([d[1], -d[0], 0.0])  # outward normal of triangle t (CCW -> right side)
        U = lambda z: np.array([pts[u][0], pts[u][1], z])
        V = lambda z: np.array([pts[v][0], pts[v][1], z])
        if len(lst) == 1:
            quad(U(bot_z(u, t)), V(bot_z(v, t)), V(top_z(v, t)), U(top_z(u, t)), out_a)
        else:
            t2 = lst[1][0]
            # top step (rim of the inset disc)
            if in_well[t] != in_well[t2]:
                if in_well[t2]:
                    quad(U(top_z(u, t2)), V(top_z(v, t2)), V(top_z(v, t)), U(top_z(u, t)), out_a)
                else:
                    quad(U(top_z(u, t)), V(top_z(v, t)), V(top_z(v, t2)), U(top_z(u, t2)), -out_a)
            # bottom step
            ba_u, ba_v = bot_z(u, t), bot_z(v, t)
            bb_u, bb_v = bot_z(u, t2), bot_z(v, t2)
            if abs(ba_u - bb_u) > 1e-9 or abs(ba_v - bb_v) > 1e-9:
                if ba_u < bb_u:
                    quad(U(ba_u), V(ba_v), V(bb_v), U(bb_u), out_a)
                else:
                    quad(U(bb_u), V(bb_v), V(ba_v), U(ba_u), -out_a)

    # speaker underside (platform plane) must clear the flare everywhere outside the platform
    off = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > p["disc_d"] / 2 + 1e-6
    plane = np.array([zd(y, p) for y in pts[:, 1]])
    info = dict(min_speaker_clearance=float((plane - top)[off].min()),
                rear_edge_clearance=float(zd(D, p) - p["edge_t"]),
                min_thickness=float(min(top_z(i, t) - bot_z(i, t) for t, tr in enumerate(tri) for i in tr)))
    return np.array(tris, dtype=np.float64), info


if __name__ == "__main__":
    out = Path(__file__).with_name("kef_lsx_ii_lt_pad.stl")
    tris, info = build()
    write_stl(out, tris, name=b"KEF LSX II LT pad")
    rep = check(tris)
    print(out, out.stat().st_size, "bytes")
    for k, v in {**rep, **info}.items():
        print(f"{k}: {round(v, 2) if isinstance(v, float) else v}")
    rd = P["disc_d"] / 2
    print("platform height (mm): front", round(zd(P["disc_y"] - rd), 2), " centre", round(zd(P["disc_y"]), 2),
          " rear", round(zd(P["disc_y"] + rd), 2))
    ok = rep["bad_edges"] == 0 and rep["volume_cm3"] > 0 and info["min_speaker_clearance"] >= 0
    sys.exit(0 if ok else 1)
