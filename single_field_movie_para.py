# --- Parallel, single-core-per-file pipeline ---
# - One process per file; no intra-process threading (MKL/BLAS/OpenMP set to 1).
# - KD-tree query uses workers=1 to guarantee single-core per job.
# - Final stacking is sequential.
# - Safe for HDF5: each process opens its own file handles.
# - Configure MAX_PROCS to <= number of physical/logical cores you want to use.

import os
# --- force single-threaded math before importing numpy/scipy ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob, gc, json, time
import numpy as np
import h5py
from scipy import io as sio
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------
# Config
# ----------------------------
fields = ['p','T','HRR','vfrac','Z','mach','rho','pressure_grad','Ux','Uy','Uz']
field = 'T'
assert field in fields

BASE_DATA_PATH = '/mnt/LaCie/RDE/SHRED_data/AFRL_RDRE'
FOLDERS = ['1309','1319','1329','1339','1349','1359','1369']

OUT_DIR = 'interpolated_quickly_para'
COMBINED_NAME = 'interpolated_combined.npy'

SUBSAMPLE_ENABLED = False
SUBSAMPLE_N = 50_000_000
SUBSAMPLE_SEED = 42

K_NEIGHBORS = 4 # 8 for the smallest grid it was 8  # Increased from 4 for more smoothing
LEAFSIZE = 40
INTERP_CHUNK = 50_000_000

GRID_PATH = 'cylindrical_grid.npy'  # expects dict with X_grid, Y_grid, Z_grid

# Upper bound on parallel workers; reasonable limit considering I/O bottlenecks
MAX_PROCS = 32  # Start with ~1/6 of cores, tune based on performance

# ----------------------------
# Utils
# ----------------------------
def now(): return time.perf_counter()

def contiguous_block_indices(N, k, seed=42):
    if k is None or k >= N:
        return np.arange(N, dtype=np.int64)
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, N - k)))
    return np.arange(start, start + k, dtype=np.int64)

def normalize_vec_shape(shape):
    if len(shape) == 1: return ('1d', None)
    if len(shape) == 2:
        if shape[1] == 1: return ('2d_col', 0)
        if shape[0] == 1: return ('2d_row', 1)
    raise ValueError(f'Expected vector-like dataset, got shape={shape}')

def vec_length(ds):
    kind, _ = normalize_vec_shape(ds.shape)
    if kind == '1d': return ds.shape[0]
    if kind == '2d_col': return ds.shape[0]
    if kind == '2d_row': return ds.shape[1]

def slice_vec_h5(ds, idx_sorted):
    kind, _ = normalize_vec_shape(ds.shape)
    if kind == '1d':
        return np.asarray(ds[idx_sorted], dtype=np.float32).ravel()
    if kind == '2d_col':
        return np.asarray(ds[idx_sorted, 0], dtype=np.float32).ravel()
    if kind == '2d_row':
        return np.asarray(ds[0, idx_sorted], dtype=np.float32).ravel()

def find_dataset_paths_h5(h5file, names_lower):
    wanted = {n.lower(): n for n in names_lower}
    found = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            lname = name.split('/')[-1].lower()
            if lname in wanted and wanted[lname] not in found:
                found[wanted[lname]] = name
    h5file.visititems(visit)
    return found

def load_rows_hdf5(mat_path, wanted=('x','y','z'), value_var=None,
                   subsample_enabled=True, subsample_n=200_000, seed=0):
    all_vars = list(wanted)
    if value_var is not None and value_var not in all_vars:
        all_vars.append(value_var)
    with h5py.File(mat_path, 'r') as f:
        paths = find_dataset_paths_h5(f, [v.lower() for v in all_vars])
        if set(all_vars) - set(paths.keys()):
            missing = list(set(all_vars) - set(paths.keys()))
            raise RuntimeError(f'Missing datasets {missing}')
        ref = 'x' if 'x' in paths else all_vars[0]
        N = vec_length(f[paths[ref]])
        idx = contiguous_block_indices(N, subsample_n if subsample_enabled else None, seed=seed)
        out = {}
        for v in all_vars:
            out[v] = slice_vec_h5(f[paths[v]], idx)
    return out, idx, N, True  # partial_io=True

def load_rows_legacy(mat_path, wanted=('x','y','z'), value_var=None,
                     subsample_enabled=True, subsample_n=200_000, seed=0):
    all_vars = list(wanted)
    if value_var is not None and value_var not in all_vars:
        all_vars.append(value_var)
    data = sio.loadmat(mat_path, variable_names=tuple(all_vars))
    def pull(d, key):
        for k in d.keys():
            if k.lower() == key.lower():
                return np.asarray(d[k]).ravel()
        raise KeyError(key)
    x_all = pull(data, 'x'); y_all = pull(data, 'y'); z_all = pull(data, 'z')
    N = x_all.size
    idx = contiguous_block_indices(N, subsample_n if subsample_enabled else None, seed=seed)
    out = {
        'x': x_all[idx].astype(np.float32, copy=False),
        'y': y_all[idx].astype(np.float32, copy=False),
        'z': z_all[idx].astype(np.float32, copy=False),
    }
    if value_var is not None:
        out[value_var] = pull(data, value_var)[idx].astype(np.float32, copy=False)
    del data, x_all, y_all, z_all
    gc.collect()
    return out, idx, N, False  # partial_io=False

def load_rows_any(mat_path, wanted=('x','y','z'), value_var=None,
                  subsample_enabled=True, subsample_n=200_000, seed=0):
    try:
        return load_rows_hdf5(mat_path, wanted, value_var, subsample_enabled, subsample_n, seed)
    except Exception:
        return load_rows_legacy(mat_path, wanted, value_var, subsample_enabled, subsample_n, seed)

def get_array_from_dict(d, names):
    for n in names:
        for k in d.keys():
            if k.lower() == n.lower():
                return np.asarray(d[k]).ravel()
    return None

def query_kdtree_singlecore(tree, q, k):
    # Force single-threaded query inside each process
    try:
        d, idx = tree.query(q, k=k, workers=1)
    except TypeError:
        d, idx = tree.query(q, k=k)
    if k == 1 or (np.ndim(d) == 1):
        d = d[:, None]; idx = idx[:, None]
    return d, idx

# ----------------------------
# Load cylindrical grid (once; inherited by forked workers)
# ----------------------------
cyl = np.load(GRID_PATH, allow_pickle=True).item()
Xg = get_array_from_dict(cyl, ['X_grid']); Yg = get_array_from_dict(cyl, ['Y_grid']); Zg = get_array_from_dict(cyl, ['Z_grid'])
if Xg is None or Yg is None or Zg is None:
    raise ValueError("Grid must contain X_grid, Y_grid, Z_grid")
Xg = Xg.astype(np.float32, copy=False).ravel()
Yg = Yg.astype(np.float32, copy=False).ravel()
Zg = Zg.astype(np.float32, copy=False).ravel()
if not (Xg.size == Yg.size == Zg.size):
    raise ValueError("Grid arrays must be equal length")
grid_points = np.vstack((Xg, Yg, Zg)).T
N_GRID = grid_points.shape[0]
del cyl, Xg, Yg, Zg
gc.collect()
print(f'Grid points: {N_GRID}')

# ----------------------------
# Collect files
# ----------------------------
mat_files = []
for folder in FOLDERS:
    p = os.path.join(BASE_DATA_PATH, folder)
    if os.path.isdir(p):
        mat_files.extend(glob.glob(os.path.join(p, '*.mat')))
mat_files = sorted(mat_files)
print(f'Files: {len(mat_files)}')

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Worker: process single file
# ----------------------------
def process_one(mat_path: str) -> dict:
    base = os.path.basename(mat_path)
    t0 = now()
    try:
        sel, used_idx, N_total, partial = load_rows_any(
            mat_path,
            wanted=('x','y','z'),
            value_var=field,
            subsample_enabled=SUBSAMPLE_ENABLED,
            subsample_n=SUBSAMPLE_N,
            seed=SUBSAMPLE_SEED
        )
    except Exception as e:
        return {
            "file": base,
            "status": "error",
            "error": f"load_failed: {e}",
        }
    t1 = now()

    x = sel['x']; y = sel['y']; z = sel['z']; V = sel[field]
    if not (x.size == y.size == z.size == V.size):
        del sel
        gc.collect()
        return {
            "file": base,
            "status": "error",
            "error": "length_mismatch",
        }

    coords = np.vstack((x, y, z)).T.astype(np.float32, copy=False)
    V = V.astype(np.float32, copy=False)
    del sel, x, y, z
    gc.collect()
    t2 = now()

    tree = cKDTree(coords, leafsize=LEAFSIZE)
    t3 = now()

    out_path = os.path.join(OUT_DIR, f'interpolated_{base}.npy')
    out_mem = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N_GRID,))

    chunk = INTERP_CHUNK
    k = min(K_NEIGHBORS, coords.shape[0])
    for s in range(0, N_GRID, chunk):
        e = min(s + chunk, N_GRID)
        q = grid_points[s:e]
        d, idx = query_kdtree_singlecore(tree, q, k=k)
        # Modified weighting for more smoothing: use power of 0.5 instead of 1.0
        # This reduces the influence of distance, creating smoother interpolation
        w = 1.0 / (d**0.5 + 1e-12).astype(np.float32)
        w /= w.sum(axis=1, keepdims=True)
        vals = V[idx]
        out_mem[s:e] = (w * vals).sum(axis=1, dtype=np.float32)
        del q, d, idx, w, vals
        gc.collect()

    del out_mem, tree, coords, V
    gc.collect()
    t4 = now()

    meta = {
        "file": base,
        "partial_io": bool(partial),
        "N_total": int(N_total),
        "used_idx_start": int(used_idx[0]) if used_idx.size else None,
        "used_idx_len": int(used_idx.size),
        "subsample_enabled": bool(SUBSAMPLE_ENABLED),
        "subsample_n": int(SUBSAMPLE_N),
        "timings_sec": {
            "load": round(t1 - t0, 3),
            "prep": round(t2 - t1, 3),
            "kdtree": round(t3 - t2, 3),
            "interp_io": round(t4 - t3, 3),
            "total": round(t4 - t0, 3),
        }
    }
    meta_path = os.path.join(OUT_DIR, f'interpolated_{base}.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    return {
        "file": base,
        "status": "ok",
        "partial": bool(partial),
        "used": f'{meta["used_idx_len"]}/{meta["N_total"]}',
        "timings": meta["timings_sec"],
        "npy": out_path,
        "json": meta_path,
    }

# ----------------------------
# Parallel execution
# ----------------------------
def run_parallel(files):
    results = []
    if not files:
        return results
    
    # Determine optimal worker count
    # - Can't have more workers than files
    # - For I/O intensive tasks, optimal is often much less than CPU count
    # - Consider memory usage per worker
    optimal_workers = min(MAX_PROCS, len(files), 48)  # Cap at 48 for I/O bound tasks
    n_workers = optimal_workers
    print(f'Files to process: {len(files)}')
    print(f'Launching {n_workers} workers (optimal for I/O-bound tasks)')

    # Important on some platforms; also avoids recursive process spawning
    # and ensures globals (grid_points) are available via fork.
    start_time = now()
    active_count = 0
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut2file = {ex.submit(process_one, f): f for f in files}
        active_count = len(fut2file)
        
        for fut in as_completed(fut2file):
            try:
                r = fut.result()
            except Exception as e:
                base = os.path.basename(fut2file[fut])
                r = {"file": base, "status": "error", "error": f"worker_exception: {e}"}
            results.append(r)
            completed_count += 1
            active_count -= 1
            
            # Progress and utilization logging
            elapsed = now() - start_time
            if r.get("status") == "ok":
                t = r["timings"]
                print(f'  done {r["file"]} ({completed_count}/{len(files)}) | '
                      f'partial={r["partial"]} | used={r["used"]} | '
                      f'load={t["load"]}s prep={t["prep"]}s kdtree={t["kdtree"]}s '
                      f'interp+io={t["interp_io"]}s total={t["total"]}s | '
                      f'elapsed={elapsed:.1f}s')
            else:
                print(f'  fail {r.get("file")} ({completed_count}/{len(files)}) | {r.get("error")}')
    
    total_elapsed = now() - start_time        
    print(f'Parallel processing complete: {len(files)} files in {total_elapsed:.1f}s')
    return results

# ----------------------------
# Combine into one memmap (sequential)
# ----------------------------
def combine_outputs(npy_files, n_grid, out_dir, combined_name):
    npy_files = [f for f in npy_files if f is not None]
    npy_files = sorted(npy_files)
    npy_files = [f for f in npy_files if os.path.basename(f) != combined_name]
    if not npy_files:
        print('No per-file outputs to combine.')
        return
    combined_path = os.path.join(out_dir, combined_name)
    combined = np.lib.format.open_memmap(combined_path, mode='w+', dtype=np.float32,
                                         shape=(len(npy_files), n_grid))
    r = 0
    for f in npy_files:
        arr = np.load(f, mmap_mode='r')
        if arr.ndim != 1 or arr.size != n_grid:
            print(f'  Skip shape {arr.shape} in {os.path.basename(f)}')
            del arr; gc.collect(); continue
        combined[r, :] = arr
        del arr; gc.collect()
        r += 1
    del combined
    gc.collect()
    print(f'Combined {r} -> {combined_path}')

# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    results = run_parallel(mat_files)
    produced = [r.get("npy") if r.get("status") == "ok" else None for r in results]
    combine_outputs(produced, N_GRID, OUT_DIR, COMBINED_NAME)
