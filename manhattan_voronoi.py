"""
Manhattan Voronoi Explorer
Click grid points to switch sites on and off; the Voronoi diagram under the
chosen distance (Manhattan by default) is shaded instantly.

Run locally:   streamlit run manhattan_voronoi_app.py
Requires:      streamlit, numpy, pandas, pillow, streamlit-image-coordinates
"""

import random
import string

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Manhattan Voronoi Explorer", page_icon="🗺️", layout="wide")

# ---------------------------------------------------------------------------
# Colours: eight hues, used as light tints for regions and full strength for sites
# ---------------------------------------------------------------------------
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


SITE_RGB = np.array([hex_rgb(h) for h in HUES])
FILL_RGB = (SITE_RGB * 0.38 + 255 * 0.62).astype(np.uint8)   # light tints
TIE_LIGHT = np.array([236, 236, 232], dtype=np.uint8)
TIE_DARK = np.array([200, 200, 194], dtype=np.uint8)
GRID_RGB = np.array([255, 255, 255], dtype=np.uint8)
EMPTY_GRID_RGB = np.array([215, 214, 208], dtype=np.uint8)
BOUNDARY_RGB = np.array([40, 40, 40], dtype=np.uint8)
INK = (30, 30, 30)
MUTED = (110, 110, 105)

TARGET_PX = 620
LEFT, BOTTOM, TOP, RIGHT = 54, 44, 24, 24   # margins in pixels


def site_label(uid: int) -> str:
    letters = string.ascii_uppercase
    out = ""
    uid += 1
    while uid:
        uid, r = divmod(uid - 1, 26)
        out = letters[r] + out
    return out


def get_font(size):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("sites", {})        # (x, y) -> uid   (insertion order = creation order)
ss.setdefault("next_uid", 0)
ss.setdefault("colors", {})       # uid -> colour index
ss.setdefault("last_click", None)


def toggle_site(pt):
    if pt in ss.sites:
        uid = ss.sites.pop(pt)
        ss.colors.pop(uid, None)
    else:
        ss.sites[pt] = ss.next_uid
        ss.next_uid += 1


def clear_sites():
    ss.sites = {}
    ss.colors = {}
    ss.next_uid = 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Grid")
c1, c2 = st.sidebar.columns(2)
W = c1.number_input("Width", min_value=2, max_value=40, value=12, step=1)
H = c2.number_input("Height", min_value=2, max_value=40, value=10, step=1)

# drop sites that fell off a shrunken grid
for pt in [p for p in ss.sites if p[0] > W or p[1] > H]:
    ss.colors.pop(ss.sites.pop(pt), None)

st.sidebar.header("Distance")
metric = st.sidebar.radio(
    "Metric",
    ["Manhattan (taxicab)", "Euclidean", "Chebyshev (chessboard)"],
    help="Manhattan: |Δx| + |Δy|.  Euclidean: √(Δx² + Δy²).  Chebyshev: max(|Δx|, |Δy|).",
)
mode = st.sidebar.radio("Shade", ["Whole plane", "Grid points only"],
                        help="'Grid points only' colours each lattice point by its nearest site.")

st.sidebar.header("Show")
show_bounds = st.sidebar.checkbox("Region boundaries", value=True)
show_ties = st.sidebar.checkbox("Tie regions (hatched)", value=True,
                                help="Points exactly equidistant from two or more nearest sites.")
show_dist = st.sidebar.checkbox("Distance to nearest site at each grid point", value=False)

st.sidebar.header("Sites")
b1, b2 = st.sidebar.columns(2)
if b1.button("Clear all", width="stretch"):
    clear_sites()
if b2.button("Undo last", width="stretch") and ss.sites:
    last = max(ss.sites, key=ss.sites.get)
    ss.colors.pop(ss.sites.pop(last), None)
k = st.sidebar.slider("Random sites", 2, 15, 5)
if st.sidebar.button("Place random sites", width="stretch"):
    clear_sites()
    all_pts = [(x, y) for x in range(W + 1) for y in range(H + 1)]
    for pt in random.sample(all_pts, min(k, len(all_pts))):
        toggle_site(pt)

# ---------------------------------------------------------------------------
# Geometry (exact integer arithmetic so ties are detected exactly)
# ---------------------------------------------------------------------------
S = int(max(10, min(64, TARGET_PX // max(W, H))))   # pixels per grid unit


def dist_fn(dx, dy, metric):
    dx, dy = np.abs(dx), np.abs(dy)
    if metric.startswith("Manhattan"):
        return dx + dy
    if metric.startswith("Euclidean"):
        return dx * dx + dy * dy          # squared: same ordering, still exact
    return np.maximum(dx, dy)


@st.cache_data(max_entries=64)
def plane_labels(sites, W, H, S, metric):
    """Nearest-site index for every pixel; -1 where two or more sites tie."""
    # pixel centres in units of 1/(2S): odd integers, sites at multiples of 2S
    gx = (2 * np.arange(W * S) + 1).astype(np.int64)[None, :]
    gy = (2 * H * S - (2 * np.arange(H * S) + 1)).astype(np.int64)[:, None]
    best = np.full((H * S, W * S), np.iinfo(np.int64).max, dtype=np.int64)
    label = np.full((H * S, W * S), -1, dtype=np.int32)
    tie = np.zeros((H * S, W * S), dtype=bool)
    for i, (sx, sy) in enumerate(sites):
        d = dist_fn(gx - 2 * S * sx, gy - 2 * S * sy, metric)
        closer = d < best
        equal = d == best
        tie = np.where(closer, False, tie | equal)
        label = np.where(closer, i, label)
        best = np.minimum(best, d)
    label[tie] = -1
    return label


def lattice_info(sites, W, H, metric):
    """For every grid point: nearest site index (-1 on tie) and the true distance."""
    xs = np.arange(W + 1)[None, :]
    ys = np.arange(H + 1)[:, None]
    D = np.stack([dist_fn(xs - sx, ys - sy, metric) for sx, sy in sites])   # (n, H+1, W+1)
    dmin = D.min(axis=0)
    owners = (D == dmin).sum(axis=0)
    lab = np.where(owners == 1, D.argmin(axis=0), -1)
    true_d = np.sqrt(dmin) if metric.startswith("Euclidean") else dmin
    return lab, true_d   # row 0 is y = 0


def assign_colors(sites_uids, adj):
    """Keep each site's colour if no neighbour shares it; otherwise pick a free one."""
    colors = {}
    for i, uid in enumerate(sites_uids):
        taken = {colors[sites_uids[j]] for j in adj.get(i, ()) if sites_uids[j] in colors}
        prev = ss.colors.get(uid)
        if prev is not None and prev not in taken:
            colors[uid] = prev
        else:
            free = [c for c in range(len(HUES)) if c not in taken]
            colors[uid] = free[0] if free else i % len(HUES)
    ss.colors = colors
    return colors


def adjacency(label):
    """Regions count as neighbours if they come within 3 px (boundaries may be thin tie lines)."""
    adj = {}
    pairs = []
    for d in (1, 2, 3):
        pairs += [(label[:, :-d], label[:, d:]), (label[:-d, :], label[d:, :]),
                  (label[:-d, :-d], label[d:, d:]), (label[:-d, d:], label[d:, :-d])]
    for a, b in pairs:
        m = (a != b) & (a >= 0) & (b >= 0)
        for p, q in set(zip(a[m].tolist(), b[m].tolist())):
            adj.setdefault(p, set()).add(q)
            adj.setdefault(q, set()).add(p)
    return adj


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def render(sites, uids):
    n = len(sites)
    Wp, Hp = W * S, H * S
    img = np.full((TOP + Hp + BOTTOM, LEFT + Wp + RIGHT, 3), 255, dtype=np.uint8)
    dom = img[TOP:TOP + Hp, LEFT:LEFT + Wp]

    label = None
    colors = {}
    if n:
        label = plane_labels(tuple(sites), W, H, S, metric)
        colors = assign_colors(uids, adjacency(label))
        col_idx = np.array([colors[u] for u in uids])

        if mode == "Whole plane":
            dom[:] = FILL_RGB[col_idx[np.clip(label, 0, None)]]
            tie = label < 0
            if show_ties:
                yy, xx = np.indices(label.shape)
                stripe = ((xx + yy) // 4) % 2 == 0
                dom[tie & stripe] = TIE_DARK
                dom[tie & ~stripe] = TIE_LIGHT
            else:
                dom[tie] = TIE_LIGHT
            if show_bounds:
                edge = np.zeros_like(label, dtype=bool)
                edge[:, 1:] |= label[:, 1:] != label[:, :-1]
                edge[1:, :] |= label[1:, :] != label[:-1, :]
                dom[edge] = BOUNDARY_RGB

    # grid lines
    line_rgb = GRID_RGB if (n and mode == "Whole plane") else EMPTY_GRID_RGB
    for gx in range(W + 1):
        c = min(gx * S, Wp - 1)
        dom[:, c] = np.where((dom[:, c] == BOUNDARY_RGB).all(-1, keepdims=True), dom[:, c], line_rgb)
    for gy in range(H + 1):
        r = min(gy * S, Hp - 1)
        dom[r, :] = np.where((dom[r, :] == BOUNDARY_RGB).all(-1, keepdims=True), dom[r, :], line_rgb)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    def to_px(x, y):
        return LEFT + x * S, TOP + (H - y) * S

    # axes ticks
    f_tick = get_font(11)
    tick_gap = int(S * 0.32) + 5 if (n and mode == "Grid points only") else 8
    step = 1 if max(W, H) <= 20 else 2 if max(W, H) <= 30 else 5
    for x in range(0, W + 1, step):
        px, py = to_px(x, 0)
        draw.text((px, py + tick_gap), str(x), fill=MUTED, font=f_tick, anchor="mt")
    for y in range(0, H + 1, step):
        px, py = to_px(0, y)
        draw.text((px - tick_gap, py), str(y), fill=MUTED, font=f_tick, anchor="rm")

    # lattice points (and discrete shading / distances)
    lat_lab, lat_d = (lattice_info(sites, W, H, metric) if n else (None, None))
    r_pt = max(2, S // 12)
    r_disc = max(5, int(S * 0.3))
    f_dist = get_font(max(9, min(14, int(S * 0.3))))
    discrete = n and mode == "Grid points only"
    for x in range(W + 1):
        for y in range(H + 1):
            px, py = to_px(x, y)
            if discrete:
                li = lat_lab[y, x]
                if li >= 0:
                    c = colors[uids[li]]
                    fill, ring = tuple(FILL_RGB[c].tolist()), tuple(int(v) for v in SITE_RGB[c])
                else:
                    fill, ring = tuple(TIE_LIGHT.tolist()), MUTED
                draw.ellipse([px - r_disc, py - r_disc, px + r_disc, py + r_disc], fill=fill, outline=ring, width=2)
            else:
                draw.ellipse([px - r_pt, py - r_pt, px + r_pt, py + r_pt], fill=(150, 150, 145))
            if n and show_dist and S >= 22 and (x, y) not in ss.sites:
                d = lat_d[y, x]
                txt = f"{d:g}" if not metric.startswith("Euclidean") else f"{d:.1f}"
                if discrete:
                    draw.text((px, py), txt, fill=INK, font=f_dist, anchor="mm")
                else:
                    draw.text((px + 3, py - 3), txt, fill=INK, font=f_dist, anchor="lb",
                              stroke_width=2, stroke_fill=(255, 255, 255))

    # sites
    r_site = max(6, int(S * (0.32 if discrete else 0.22)))
    f_lab = get_font(max(11, min(16, S // 2)))
    for (x, y), uid in zip(sites, uids):
        px, py = to_px(x, y)
        fill = tuple(int(v) for v in SITE_RGB[colors[uid]])
        draw.ellipse([px - r_site - 2, py - r_site - 2, px + r_site + 2, py + r_site + 2], fill=(255, 255, 255))
        draw.ellipse([px - r_site, py - r_site, px + r_site, py + r_site], fill=fill, outline=INK, width=1)
        draw.text((px + r_site + 3, py - r_site - 1), site_label(uid), fill=INK, font=f_lab, anchor="lb",
                  stroke_width=2, stroke_fill=(255, 255, 255))
    return pil, label, colors, lat_lab


def handle_click():
    v = ss.get("voronoi_click")
    if not v:
        return
    stamp = (v.get("x"), v.get("y"), v.get("unix_time"))
    if stamp == ss.last_click:
        return
    ss.last_click = stamp
    scale = ss.img_width / v["width"] if v.get("width") else 1.0
    px, py = v["x"] * scale, v["y"] * scale
    gx = (px - LEFT) / S
    gy = H - (py - TOP) / S
    x, y = round(gx), round(gy)
    if 0 <= x <= W and 0 <= y <= H and abs(gx - x) <= 0.5 and abs(gy - y) <= 0.5:
        toggle_site((int(x), int(y)))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🗺️ Manhattan Voronoi Explorer")
st.markdown(
    "**Click any grid point** to add a site; click it again to remove it. "
    "Each shaded region is the set of points closer to that site than to any other, "
    f"using **{metric.split(' ')[0]}** distance."
)

sites = list(ss.sites.keys())
uids = list(ss.sites.values())
pil, label, colors, lat_lab = render(sites, uids)
ss.img_width = pil.width

left, right = st.columns([2.1, 1.4], gap="large")
with left:
    streamlit_image_coordinates(pil, key="voronoi_click", on_click=handle_click,
                                png_compression_level=1, cursor="pointer", width="stretch")
    if show_ties and mode == "Whole plane":
        st.caption("Hatched grey = tie: equidistant from two or more nearest sites. "
                   "Under Manhattan distance, ties can fill whole areas, not just lines.")

with right:
    if not sites:
        st.info("No sites yet. Click on the grid, or use **Place random sites** in the sidebar.")
    else:
        rows = []
        lat_counts = np.bincount(lat_lab[lat_lab >= 0].ravel(), minlength=len(sites))
        px_counts = np.bincount(label[label >= 0].ravel(), minlength=len(sites))
        for i, ((x, y), uid) in enumerate(zip(sites, uids)):
            rows.append({
                "": "●",
                "Site": site_label(uid),
                "(x, y)": f"({x}, {y})",
                "Area ≈": round(px_counts[i] / S**2, 2),
                "Grid pts": int(lat_counts[i]),
                "_c": HUES[colors[uid]],
            })
        df = pd.DataFrame(rows)
        tie_area = (label < 0).sum() / S**2
        tie_pts = int((lat_lab < 0).sum())

        styled = df.drop(columns="_c").style.apply(
            lambda col: [f"color: {c}; font-size: 1.2em" for c in df["_c"]] if col.name == "" else [""] * len(col),
            axis=0,
        )
        st.subheader("Regions")
        st.dataframe(styled, hide_index=True, width="stretch", column_config={
            "": st.column_config.TextColumn(width=30),
            "Area ≈": st.column_config.NumberColumn(format="%.2f"),
        })
        st.markdown(
            f"Ties: area ≈ **{tie_area:.2f}**, grid points **{tie_pts}**  \n"
            f"Total area {W * H}, total grid points {(W + 1) * (H + 1)}"
        )
        st.caption("Area is measured on the pixel image, so it's approximate. "
                   "Areas extend only to the edge of the grid.")

with st.expander("💡 Things to try"):
    st.markdown(
        """
- **Two sites on a horizontal line**, e.g. (3, 5) and (9, 5): the boundary is a vertical line, just like Euclidean.
- **Two sites diagonally apart**, e.g. (3, 3) and (8, 8): the boundary has a diagonal middle piece, and beyond the
  sites' square there are **whole tie regions** where both sites are equally far away.
- **Two sites slightly off-diagonal**, e.g. (3, 3) and (9, 8): the tie areas collapse into a staircase-like
  boundary with long vertical or horizontal tails.
- Switch the **metric** with the same sites and compare the shapes. Why are Euclidean boundaries always straight
  perpendicular bisectors?
- Use **Grid points only** with distances shown: count how many grid points each site "owns".
"""
    )
