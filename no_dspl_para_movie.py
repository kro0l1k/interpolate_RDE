# CPU parallel renderer: per-frame processes write PNGs; final pass encodes MP4.

import os, io, json, gc, math, time
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colorbar as colorbar
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# =========
# Config
# =========
SNAPSHOT_DIR = "loaded_snapshots"
META_JSON    = "loaded_index.json"

OUT_DIR      = "movies"
MP4_NAME     = "pointcloud_T_by_time_cpu_parallel.mp4"
FRAMES_DIR   = "frames_png"   # temp folder for PNGs

WIDTH, HEIGHT   = 1280, 960
FPS             = 10
CMAP_NAME       = "jet"
LEGEND_LABEL    = "Temperature"
ELEVATION_DEG   = 30#40.0
AZIMUTH_DEG     = 40.0
BORDER_PAD_FRAC = 0.05

MAX_POINTS_DEFAULT = 48_000_000  # per frame
RNG_SEED           = 1234

print(f'Using up to {cpu_count()} CPU cores')
MAX_WORKERS_DEFAULT = min(120, cpu_count())  # tune for disk bandwidth
VERBOSE = True

# =========
# Math utils
# =========
def rotation_matrix(elev_deg: float, azim_deg: float) -> np.ndarray:
    az = math.radians(azim_deg)
    el = math.radians(elev_deg)
    Rz = np.array([[ math.cos(az), -math.sin(az), 0.0],
                   [ math.sin(az),  math.cos(az), 0.0],
                   [ 0.0,           0.0,          1.0]], dtype=np.float32)
    Rx = np.array([[ 1.0, 0.0,           0.0          ],
                   [ 0.0, math.cos(el), -math.sin(el)],
                   [ 0.0, math.sin(el),  math.cos(el)]], dtype=np.float32)
    return Rx @ Rz

def project_points(points_xyz: np.ndarray, R: np.ndarray, center: np.ndarray):
    P = (points_xyz.astype(np.float32) - center[None, :]).T  # (3,N)
    Pr = (R @ P).T
    return Pr[:, 0], Pr[:, 1], Pr[:, 2]

# =========
# Legend
# =========
def create_legend_image(cmap_callable, vmin, vmax, label='Temperature', width=120, height=400) -> Image.Image:
    norm = Normalize(vmin=vmin, vmax=vmax)
    import matplotlib.colorbar as colorbar  # local to avoid backend issues
    fig, ax = plt.subplots(figsize=(1.2, 4))
    fig.subplots_adjust(left=0.3, right=0.7)
    cb = colorbar.ColorbarBase(ax, cmap=cmap_callable, norm=norm, orientation='vertical')
    cb.set_label(label, fontsize=10)
    ticks = np.linspace(vmin, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.0f}" for t in ticks])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGBA').resize((width, height))

# =========
# Rasterizer (nearest-in-depth)
# =========
def rasterize_nearest(xp, yp, zp, Tp, width, height, xlim, ylim):
    xmin, xmax = xlim
    ymin, ymax = ylim
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        xmax = xmin + 1.0
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        ymax = ymin + 1.0

    xn = (xp - xmin) / (xmax - xmin)
    yn = (yp - ymin) / (ymax - ymin)
    ix = np.clip((xn * (width  - 1)).astype(np.int32),  0, width  - 1)
    iy = np.clip(((1.0 - yn) * (height - 1)).astype(np.int32), 0, height - 1)

    pid = iy.astype(np.int64) * width + ix.astype(np.int64)

    order = np.lexsort((zp, pid))  # pid asc, z asc
    pid_sorted = pid[order]
    _, first_idx, counts = np.unique(pid_sorted, return_index=True, return_counts=True)
    last_idx = first_idx + counts - 1
    keep = order[last_idx]

    canvas_T = np.full((height, width), np.nan, dtype=np.float32)
    canvas_T.flat[pid[keep]] = Tp[keep].astype(np.float32)
    return canvas_T

# =========
# Global state for workers (set once in parent, copied on fork)
# =========
G = {
    "R": None,
    "center": None,
    "xlim": None,
    "ylim": None,
    "vmin": None,
    "vmax": None,
    "legend_rgba": None,
    "lut": None,
    "lut_bins": 4096,
}

def build_lut(vmin, vmax, cmap_callable, bins=4096):
    xs = np.linspace(0.0, 1.0, bins).astype(np.float32)
    cols = (cmap_callable(xs)[..., :4] * 255.0).astype(np.uint8)  # (bins,4)
    return cols

def map_T_to_rgba(canvas_T, vmin, vmax, lut, bins):
    rgba = np.zeros((canvas_T.shape[0], canvas_T.shape[1], 4), dtype=np.uint8)
    mask = np.isfinite(canvas_T)
    if not mask.any():
        return rgba
    nv = (np.clip(canvas_T[mask], vmin, vmax) - vmin) / max(vmax - vmin, 1e-12)
    idx = np.minimum((nv * (bins - 1)).astype(np.int32), bins - 1)
    rgba_vals = lut[idx]
    rgba[mask] = rgba_vals
    return rgba

# =========
# Worker
# =========
def render_one(i, m, max_points, frames_dir):
    rng = np.random.default_rng(RNG_SEED + i)
    with np.load(m['npz']) as d:
        pos = d['pos'].astype(np.float32, copy=False)
        Tv  = d['T'].astype(np.float32, copy=False)

    N = pos.shape[0]
    take = min(N, max_points)
    if take < N:
        idx = rng.choice(N, size=take, replace=False)
        P = pos[idx]; Ttake = Tv[idx]
    else:
        P = pos; Ttake = Tv

    xp, yp, zp = project_points(P, G["R"], G["center"])
    canvas_T = rasterize_nearest(xp, yp, zp, Ttake, WIDTH, HEIGHT, G["xlim"], G["ylim"])
    rgba = map_T_to_rgba(canvas_T, G["vmin"], G["vmax"], G["lut"], G["lut_bins"])

    frame = Image.fromarray(rgba, mode='RGBA')
    legend_img = Image.fromarray(G["legend_rgba"], mode='RGBA')
    frame.paste(legend_img, (WIDTH - legend_img.width - 10, (HEIGHT - legend_img.height)//2), legend_img)

    out_path = os.path.join(frames_dir, f"{i:06d}.png")
    frame.save(out_path, format="PNG", compress_level=1)

    del pos, Tv, P, Ttake, xp, yp, zp, canvas_T, rgba, frame
    gc.collect()
    return out_path

# =========
# Orchestrator
# =========
def render_movie_parallel(manifest, out_dir, mp4_name, max_workers=None, max_points=None):
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, FRAMES_DIR)
    os.makedirs(frames_dir, exist_ok=True)

    if max_workers is None:
        max_workers = int(os.environ.get("RENDER_MAX_WORKERS", str(MAX_WORKERS_DEFAULT)))
    if max_points is None:
        max_points = int(os.environ.get("RENDER_MAX_POINTS", str(MAX_POINTS_DEFAULT)))

    # Probe first frame
    d0 = np.load(manifest[0]['npz'])
    pos0 = d0['pos'].astype(np.float32, copy=False)
    T0   = d0['T'].astype(np.float32, copy=False)

    # Global color limits
    vmins, vmaxs = [], []
    for m in manifest:
        with np.load(m['npz']) as d:
            a = d['T']
            vmins.append(np.nanmin(a))
            vmaxs.append(np.nanmax(a))
    vmin = float(np.nanmin(vmins)); vmax = float(np.nanmax(vmaxs))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1e-12

    # Camera
    R = rotation_matrix(ELEVATION_DEG, AZIMUTH_DEG)
    center = pos0.mean(axis=0)
    x0, y0, _ = project_points(pos0, R, center)
    xr = (np.nanmin(x0), np.nanmax(x0))
    yr = (np.nanmin(y0), np.nanmax(y0))
    xpad = BORDER_PAD_FRAC * (xr[1] - xr[0] + 1e-12)
    ypad = BORDER_PAD_FRAC * (yr[1] - yr[0] + 1e-12)
    xlim = (xr[0] - xpad, xr[1] + xpad)
    ylim = (yr[0] - ypad, yr[1] + ypad)

    # Colormap + legend
    from matplotlib import colormaps
    cmap_callable = colormaps.get_cmap(CMAP_NAME)
    legend_img = create_legend_image(cmap_callable, vmin, vmax, label=LEGEND_LABEL, width=120, height=400)
    legend_rgba = np.array(legend_img, dtype=np.uint8)

    # LUT
    lut_bins = G["lut_bins"]
    lut = build_lut(vmin, vmax, cmap_callable, bins=lut_bins)

    # Share into globals (copied to workers on fork)
    G["R"] = R; G["center"] = center; G["xlim"] = xlim; G["ylim"] = ylim
    G["vmin"] = vmin; G["vmax"] = vmax
    G["legend_rgba"] = legend_rgba
    G["lut"] = lut

    if VERBOSE:
        print(f'[SETUP] frames={len(manifest)}  workers={max_workers}  max_points/frame={max_points}')
        print(f'[SETUP] vmin={vmin:.6g} vmax={vmax:.6g}  xlim={xlim} ylim={ylim}')

    # Parallel render → PNGs
    start = time.time()
    paths = [None] * len(manifest)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(render_one, i, m, max_points, frames_dir): i for i, m in enumerate(manifest)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                outp = fut.result()
                paths[i] = outp
                done += 1
                if VERBOSE and (done % 10 == 0 or done == len(manifest)):
                    print(f'  rendered {done}/{len(manifest)}')
            except Exception as e:
                print(f'  frame {i} failed: {e}')
    if VERBOSE:
        print(f'[RENDER] PNGs complete in {time.time()-start:.1f}s')

    # Encode MP4 (sequential, stream PNGs)
    mp4_path = os.path.join(out_dir, mp4_name)
    writer = imageio.get_writer(mp4_path, fps=FPS, codec="libx264", quality=8)
    for p in paths:
        if p is None:
            raise RuntimeError('missing frame PNG')
        img = imageio.v2.imread(p)
        writer.append_data(img)
    writer.close()
    if VERBOSE:
        print(f'[DONE] {mp4_path}')

# =========
# Main (render-only; assume manifest present)
# =========
if __name__ == "__main__":
    manifest_path = os.path.join(SNAPSHOT_DIR, META_JSON)
    with open(manifest_path, "r") as fh:
        manifest = json.load(fh)["frames"]
    assert len(manifest) > 0

    render_movie_parallel(
        manifest,
        OUT_DIR,
        MP4_NAME,
        max_workers=int(os.environ.get("RENDER_MAX_WORKERS", str(min(64, cpu_count())))),
        max_points=int(os.environ.get("RENDER_MAX_POINTS", str(MAX_POINTS_DEFAULT))),
    )
