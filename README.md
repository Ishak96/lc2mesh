# lc2mesh — asteroid shape reconstruction from lightcurves
## Technical University of Denmark Team Submission

`lc2mesh` reconstructs the 3D shape of an asteroid from its observed
lightcurves. It fits **one implicit neural representation (INR) per asteroid**:
a coordinate SIREN deforms an ellipsoid prior, a graph CNN then refines that
mesh, and both are optimized by comparing the lightcurves produced by a
differentiable photometric forward model against the real measured curves.
There is no pretrained lightcurve encoder and no 3D ground truth in the
objective — the true STL is used **for evaluation only**, when available.

## How it works

**Stage 1 — coarse INR (convex physics).** A SIREN predicts a log-radial field
over an ellipsoid icosphere prior (`v -> v * exp(field)`), canonicalized into the
challenge cylinder `D(0, R) x [-1, 1]`. The loss is the mean-normalized
lightcurve MSE on the training cameras plus a Chamfer term anchoring the mesh to
the convex hull of an external convex-inversion prior, plus a one-ring Laplacian
on the displacement field.

**Stage 2 — graph-CNN refinement (self-occlusion physics).** A fixed-topology
graph CNN predicts per-vertex displacements. A convex forward model is blind to
concavities, so Stage 2 multiplies a ray-cast sun/observer visibility mask onto
the forward matrix: a concavity shadows its own facets, which gives the refiner a
gradient for *where* to carve. Edge-length, Laplacian and hull-containment terms
keep the deformation smooth and inside the convex hull.

**Convex-floor acceptance gate.** On a nearly convex body the occlusion physics
adds nothing, and Stage 2 can only chase an unfittable residual. The Stage-1 mesh
is therefore scored under the convex model (the *convex floor*) before Stage 2;
afterwards the refined mesh is re-scored under occlusion physics with a freshly
ray-cast mask. Stage 2 is kept only if it beats the floor; otherwise the Stage-1
state is restored and everything downstream (checkpoint, exported STL, metrics)
uses the Stage-1 mesh.

## Repository structure

```
lc2mesh/
├── run.py                      # main entry point: load -> train -> reconstruct -> evaluate -> save
├── run.sh                      # thin launcher: ./run.sh <GPU_ID> <ASTEROID_ID>
├── visualize.py                # inspect a saved reconstruction: metrics, plots, rotating video
├── pyproject.toml              # Python dependencies / package metadata
├── bin/get_challenge_data.sh   # downloads the challenge dataset into $DSDIR
├── src/lc2mesh/
│   ├── config.py               # every hyperparameter of the experiment (single source of truth)
│   ├── constants.py            # observer geometry, per-asteroid cylinder radii, data location
│   ├── data.py                 # challenge lightcurve loading
│   ├── prior.py                # ellipsoid/icosphere prior mesh construction
│   ├── model.py                # SIREN INR, graph convolutions, LightcurveToMesh (INR + GCN)
│   ├── forward_model.py        # differentiable Lommel-Seeliger + Lambert forward operator
│   ├── pipeline.py             # mesh -> lightcurve pipeline incl. the ray-cast occlusion mask
│   ├── training.py             # two-stage optimization loop, losses, camera split, checkpointing
│   ├── metrics.py              # lightcurve metrics + surface Chamfer / solid overlap
│   ├── eval.py                 # official challenge voxel measures
│   ├── mesh.py                 # challenge-cylinder mesh normalization
│   ├── utils.py                # seeding, device selection, hull sampling, regularizers
│   └── visualization.py        # lightcurve plots, interactive meshes, rotating mp4 renderer
├── convinv/                    # CODE that generates the convex inversions (optional step)
│   ├── convexinitial.py        # per-asteroid driver: fit areas -> Minkowski -> STL
│   ├── convexinitial.sh        # launcher (also submittable to LSF)
│   ├── convex_inversion_multi.py  # the convex lightcurve inversion itself
│   ├── convex_forward.py       # observer/sun unit vectors
│   └── minkowski.py            # Minkowski reconstruction (areas + normals -> polyhedron)
├── convex_inversions/          # RESULTS of convinv/: one prior STL per asteroid (shipped)
├── recon/                      # OUTPUT: reconstructed meshes, reconstructed_ast<ID>.stl
├── videos/                     # OUTPUT: rotating mp4 renders of the reconstructions
├── results/                    # OUTPUT: metrics.json, per-camera CSV and figures per asteroid
└── ckpt/                       # OUTPUT: PyTorch checkpoints (git-ignored, large)
```

`convinv/` is the **code** that produces the convex-inversion priors;
`convex_inversions/` holds the **generated results**. The latter is an *input*
to the INR pipeline: `run.py` requires
`convex_inversions/asteroid<ID>_simulated_intensity.stl` as the geometric prior
(its convex hull is the Stage-1 Chamfer target and the Stage-2 containment
region). The ten priors are shipped with the repository, so `convinv/` does not
normally need to be run — see [Convex inversions](#convex-inversions).

## Requirements

- **Python** >= 3.11 (developed on 3.13).
- **PyTorch** >= 2.5. A CUDA build is strongly recommended; install the wheel
  that matches your driver (see <https://pytorch.org/get-started/locally/>).
  The code also runs on CPU, but a full run then takes hours instead of minutes.
- **GPU**: one CUDA device. At the default resolution (`N_SUBDIVISIONS = 4`,
  5120 facets, 28 cameras, 842 phases) a run peaks at roughly 9 GiB of GPU
  memory. Reduce `N_SUBDIVISIONS` in `src/lc2mesh/config.py` if you have less.
- **Embree** ray tracing through the `embreex` wheel (pulled in automatically).
  The pure-Python trimesh ray backend is ~100x too slow for the occlusion mask.
- **ffmpeg** for the mp4 renders — provided by the `imageio-ffmpeg` wheel, no
  system install required.
- **wget** and **unzip** — only needed by `bin/get_challenge_data.sh`.
- **Challenge data**: the HAC-2026 dataset (lightcurves + public shape models).
- *Optional, only to regenerate the convex inversions*: the `convinv` extra
  (`pyvista`, `numpy-stl`) and, for GPU acceleration, a CuPy wheel matching your
  CUDA runtime. No GPU is required for that step — it falls back to NumPy.

## Installation

```bash
git clone <repository-url> lc2mesh
cd lc2mesh
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Or, with [uv](https://docs.astral.sh/uv/) (this creates `.venv/` for you, which
is what `convinv/convexinitial.sh` picks up automatically):

```bash
uv sync                      # INR pipeline only
uv sync --extra convinv      # + the convex-inversion regeneration dependencies
```

If you need a specific CUDA build of PyTorch, install it first, then
`pip install -e .`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

### Challenge data

The dataset location is resolved from the `DSDIR` environment variable; without
it, `./challenge_data` inside the repository is used.

```bash
export DSDIR=/path/to/data      # optional; the script appends challenge_data/
./bin/get_challenge_data.sh     # downloads and extracts the archive
```

Expected layout (per asteroid):

```
$DSDIR/challenge_data/AsteroidModel03_shape_public/
├── Asteroid3_lightcurve_data/Asteroid03_lightcurve_intensity.txt
└── asteroid3.stl                     # public ground truth (asteroids 1-3 only)
```

Asteroids 1-3 are public and ship with a true STL; 4-10 are secret, so shape
metrics are skipped for them (the lightcurve metrics are still computed).

## Convex inversions

The repository provides **pre-generated convex inversion results** in
`convex_inversions/`. These results are used by the reconstruction pipeline and
are included to allow the experiments to be reproduced without having to
regenerate the convex inversions first. **You normally do not need to regenerate
them** — `./run.sh <GPU_ID> <ASTEROID_ID>` works out of the box. Regeneration is
only necessary if you want to reproduce the convex inversion step itself.

The priors are produced by a classical convex lightcurve inversion (the code in
`convinv/`): facet areas of a 1280-facet icosphere are fitted to the *simulated*
intensity lightcurves with a Lommel-Seeliger + Lambert forward model
(`LSLparam = 0.8`), a convexity/closure regularizer (`lambda = 0.2`, doubled and
refitted while the leftover "dark area" exceeds 1%) and an area-scale penalty,
using SciPy CG on `log(area)`; the fitted areas + normals are then turned back
into a convex polyhedron by Minkowski minimisation (`epsilon = 0.005`,
`max_iter = 20000`) and rescaled into the challenge cylinder.

### Regenerating them

1. **Environment.** Use the project's uv-managed virtual environment, including
   the optional convex-inversion dependencies (`pyvista`, `numpy-stl`):

   ```bash
   cd lc2mesh
   uv sync --extra convinv
   ```

   (Equivalent without uv: `python -m venv .venv && source .venv/bin/activate &&
   pip install -e ".[convinv]"`.) No `module load cuda/...` is needed — the
   environment supplies everything.

   GPU acceleration is **optional**: install a CuPy wheel matching your CUDA
   runtime (`uv pip install cupy-cuda12x`, or `cupy-cuda13x`) and the area fit
   runs on the GPU. Without CuPy — or on a node without a usable device — the
   code automatically falls back to the NumPy CPU backend and produces the same
   result, just slower.

2. **Run it**, from anywhere (the script resolves the repository from its own
   location):

   ```bash
   ./convinv/convexinitial.sh <ASTEROID_ID> [ORIGIN] [--force]
   ./convinv/convexinitial.sh 3 simulated
   ```

   - `<ASTEROID_ID>` — integer in `1..10` (required).
   - `[ORIGIN]` — `simulated` (default, and what the INR pipeline consumes) or
     `real`.
   - `--force` — overwrite an existing result. **Without it the script refuses
     to touch an existing STL**, so the shipped priors cannot be clobbered by
     accident.
   - `PYTHON=/path/to/python ./convinv/convexinitial.sh ...` selects a specific
     interpreter; otherwise `.venv/bin/python`, then `uv run`, then `python3` is
     used.

   Equivalent direct call, from the repository root:
   `python -m convinv.convexinitial 3 simulated`.

3. **Output.** Results are written to `convex_inversions/`:

   | File | Contents |
   | --- | --- |
   | `asteroid<ID>_<origin>_intensity.stl` | The convex prior mesh, in the challenge cylinder frame. This exact filename is what `run.py` loads. |
   | `asteroid<ID>_<origin>_intensity_info.txt` | Dark-area percentage, the regularization parameter that was accepted, the pre-normalization Minkowski radius, the true cylinder radius, the LSL parameter and the optimizer. |

4. **Runtime.** One asteroid is a CG fit of 1280 facet areas with up to 30000
   iterations, plus the Minkowski iteration — minutes on a GPU, longer on CPU,
   and longer still when the dark-area check forces a refit at a larger
   regularization. It needs no more than a few GB of memory.

**Cluster (LSF) submission.** `convinv/convexinitial.sh` keeps its original
`#BSUB` headers, so it can still be submitted as a job array over asteroids
4-10, in which case `$LSB_JOBINDEX` supplies the asteroid id:

```bash
export CONVINV_REPO_ROOT=/absolute/path/to/lc2mesh
bsub < convinv/convexinitial.sh
```

`CONVINV_REPO_ROOT` is required there because LSF copies the submitted script
into a spool directory, so the script cannot locate the repository from its own
path. The queue name, walltime and resource requests in the headers are specific
to the original cluster and must be adapted to yours.

## Running the reconstruction

```bash
./run.sh <GPU_ID> <ASTEROID_ID>
```

- `<GPU_ID>` — CUDA device index, exactly as reported by `nvidia-smi`. It is
  passed to PyTorch as `cuda:<GPU_ID>`; `CUDA_VISIBLE_DEVICES` is deliberately
  left untouched so the index keeps its usual meaning. If the index does not
  exist, the run falls back to CPU with a warning.
- `<ASTEROID_ID>` — challenge asteroid, an integer in `1..10`.

Example — train asteroid 3 on GPU 0:

```bash
./run.sh 0 3
```

Extra arguments are forwarded to `run.py`, e.g. to redirect the outputs:

```bash
./run.sh 0 3 --output-dir /scratch/lc2mesh_runs
```

`run.py` can also be called directly: `python run.py --gpu 0 --asteroid 3`.
A full run is 2000 Stage-1 steps + 2000 Stage-2 steps and takes a few minutes
per stage on a V100-class GPU.

## Output files

For `--asteroid <ID>` (all paths relative to `--output-dir`, default: the
repository root):

| Path | Contents |
| --- | --- |
| `recon/reconstructed_ast<ID>.stl` | **The reconstruction.** Binary STL, triangle mesh in the challenge cylinder frame (radius `R` from `MODEL_BASE_RADII`, `z ∈ [-1, 1]`). This is the mesh selected by the acceptance gate — refined if accepted, Stage-1 otherwise. |
| `ckpt/inr_forward_train_<ID>.pth` | PyTorch checkpoint: model weights, prior faces/vertices, the full configuration, both stage histories, the gate decision and all fit numbers. Inference artifact, not an optimizer-resume checkpoint. |
| `results/ast<ID>/metrics.json` | All scalar results of the run (see below). |
| `results/ast<ID>/per_camera_metrics.csv` | Per-camera MSE, MAE, Pearson r and the train/validation split label. |
| `results/ast<ID>/training_curves.png` | Training objective, camera-split MSE and Pearson r across both stages. |
| `results/ast<ID>/lightcurve_comparison.png` | Observed vs predicted curves, all cameras concatenated, with a zoom on one camera. |
| `results/ast<ID>/camera_grid.png` | Observed vs predicted curves, one panel per camera. |
| `videos/*_ast<ID>.mp4` | Rotating render, written by `visualize.py`. Named `true_vs_optimized_ast<ID>.mp4` when the true STL is available, `reconstruction_ast<ID>.mp4` otherwise. |

Console output mirrors the notebook: camera split, parameter counts, gradient
sanity check, per-stage progress, the gate decision, final MSEs and the voxel
measures.

## Evaluation and metrics

Evaluation happens automatically at the end of `run.py`; nothing has to be run
separately. Metrics land in `results/ast<ID>/metrics.json`.

**Lightcurve fit** (always computed, no ground truth needed)

- `final_lightcurve_mse.train` / `.validation` — MSE of the mean-normalized
  curves on the 21 training and 7 held-out validation cameras.
- `pearson_r.mean` / `.min` — per-camera Pearson correlation, all 28 cameras.
- `convex_floor`, `stage2_fit`, `stage1_under_occlusion` — the three fits the
  acceptance gate compares, each as `{train, validation, mean}`.
- `stage2_accepted`, `final_mesh_label` — the gate outcome.

**Shape quality** (only when the true STL is available, i.e. asteroids 1-3),
reported for the convex prior, the Stage-1 mesh and the final mesh, all
cylinder-normalized into the same frame:

- `measure1_one_minus_iou` = `1 - |A ∩ B| / |A ∪ B|` on filled voxels — the
  official challenge measure. **0 = identical, lower is better.**
- `measure2_symmetric_difference` = `(|A\B| + |B\A|) / (|A| + |B|)` — the second
  official measure, same convention.
- `surface_chamfer` — sum of the two mean nearest-neighbour surface distances
  between 8192 samples per mesh.

Both voxel measures are computed on a *shared, aligned* voxel grid (pitch
`0.05`), which makes them translation-sensitive. The comparison against the
convex prior and the Stage-1 mesh is what shows whether the refinement helped.

To re-score an existing STL without retraining, use `visualize.py`, which prints
the same voxel measures for `recon/reconstructed_ast<ID>.stl`.

## Visualization

```bash
python visualize.py --asteroid 3 [--gpu 0] [--html] [--no-video]
```

`visualize.py` loads `recon/reconstructed_ast<ID>.stl`, re-runs the convex
forward model on it, and produces:

- printed per-camera fit table and the official voxel measures vs the true STL,
- `results/ast<ID>/visualize_lightcurve_comparison.png` and
  `visualize_camera_grid.png`,
- `videos/true_vs_optimized_ast<ID>.mp4` — a 60-frame, 10 fps rotating render;
  two panels (target + reconstruction) when the true STL is available, one
  otherwise. Use `--no-video` to skip it,
- with `--html`, interactive Plotly meshes as
  `results/ast<ID>/reconstruction_ast<ID>.html` (rotate/zoom in a browser).

The plotting helpers in `src/lc2mesh/visualization.py`
(`plot_interactive_mesh`, `plot_lightcurve_comparison`, `plot_camera_grid`,
`render_rotating_mesh_video`) can also be imported directly from a notebook.

Note that `visualize.py` reproduces the standalone viewer settings of the
original project: `float64` arithmetic and a plain **convex** forward model with
`lambert_weight = 0.3`. Its lightcurve numbers therefore differ slightly from the
ones printed by `run.py`, which uses the training physics (`float32`,
`LAMBERT_WEIGHT = 0.5`, occlusion mask when the refinement was accepted).

## Reproducibility

- All hyperparameters live in **`src/lc2mesh/config.py`**; `run.py` reads them
  and takes only the asteroid id and the GPU id from the command line. Changing
  a value there changes the experiment — the defaults are the tuned ones.
- Seeds: `SEED = 42` (NumPy/torch/random, model init, hull sampling) and
  `SPLIT_SEED = 1729` (camera split). `run.py` sets
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`
  before touching CUDA.
- Camera split: a fixed 21/7 camera-level split (`VALIDATION_FRACTION = 0.25`).
  Whole rotation sequences stay together, and validation never enters a gradient.
  It is *conditional on the external convex-inversion prior*, so it is a
  validation set, not an independent test set.
- Key defaults: `N_SUBDIVISIONS = 4` (5120 faces), INR 128x4 with `COARSE_W0 = 10`,
  plain GCN 128x6, `LAMBERT_WEIGHT = 0.5`, 2000 + 2000 steps,
  `LR_INR_STAGE1 = 1e-4`, `LR_GRAPH = 1e-3`, `LR_INR_STAGE2 = 1e-6`,
  Stage-1 `(λ_hull, λ_lap) = (0.3, 0.1)`, Stage-2
  `(λ_hull, λ_edge, λ_lap, λ_containment) = (0.0, 0.01, 0.2, 0.1)`,
  `SELECTION_STAGE1 = "training"`, `SELECTION_STAGE2 = "last"`,
  `STAGE2_ACCEPTANCE = "mean"`, occlusion mask refreshed every 100 calls on a
  24-phase grid.
- Every checkpoint embeds the exact configuration used to produce it
  (`checkpoint["config"]`), including both `TrainingConfig` dataclasses and the
  camera split, so a run can always be traced back.
- Exact bit-for-bit reproducibility still depends on the GPU model, the CUDA
  version and the PyTorch build; the metrics are stable across runs on the same
  machine.

**Provenance caveat.** The prior in `convex_inversions/` comes from an existing
*simulated-lightcurve* convex inversion, while the targets are the real measured
lightcurves. Its camera provenance is unverified, so a better lightcurve fit does
not by itself establish better geometry.

## Troubleshooting

**`Permission denied: ./run.sh`** — `chmod +x run.sh`, or run
`bash run.sh <GPU_ID> <ASTEROID_ID>`.

**`Device: cpu` although a GPU exists** — the requested index is out of range;
`setup_device` prints the number of visible devices. Check `nvidia-smi` and pass
a valid index. If you exported `CUDA_VISIBLE_DEVICES`, the indices are
*remapped*: with `CUDA_VISIBLE_DEVICES=5`, the device to request is `0`.

**`Error: ASTEROID_ID must be in 1-10`** — only the ten challenge asteroids
exist. `run.py` rejects anything else via `--asteroid` choices.

**`FileNotFoundError: Model data directory not found ...`** — the challenge data
is missing or `DSDIR` points somewhere else. Run `./bin/get_challenge_data.sh`
and verify `$DSDIR/challenge_data/AsteroidModel0<ID>_shape_*` exists.

**`FileNotFoundError: Explicit geometric prior required: .../convex_inversions/asteroid<ID>_simulated_intensity.stl`**
— the convex-inversion prior for that asteroid is missing. It is required; the
pipeline deliberately refuses to silently fall back to a lightcurve-only fit.
The shipped priors cover asteroids 1-10; regenerate a missing one with
`./convinv/convexinitial.sh <ASTEROID_ID> simulated`.

**`ModuleNotFoundError: No module named 'pyvista'` when running `convinv/`** —
the convex-inversion extra is not installed: `uv sync --extra convinv` (or
`pip install -e ".[convinv]"`). It is not needed by the INR pipeline.

**`<path> already exists — refusing to overwrite`** — intended: the shipped
convex inversions are protected. Pass `--force` to regenerate, or `--out-dir` to
write somewhere else.

**`Error: could not locate the lc2mesh repository`** — you are running
`convexinitial.sh` from a copy (typically an LSF spool directory). Export
`CONVINV_REPO_ROOT=/absolute/path/to/lc2mesh` before submitting.

**`ModuleNotFoundError: No module named 'lc2mesh'`** — the package is not
installed in the active interpreter. Re-run `pip install -e .` inside the
virtual environment you launch `run.sh` from, or set `PYTHONPATH=src`. Use
`PYTHON=/path/to/python ./run.sh ...` to pick a specific interpreter.

**`ModuleNotFoundError: No module named 'embreex'` / very slow Stage 2** — the
Embree backend is missing, so `trimesh.ray.ray_pyembree` cannot be imported.
`pip install embreex`.

**CUDA out of memory** — lower `N_SUBDIVISIONS` (4 -> 3 quarters the facet
count) or `OCCLUSION_N_PHASES` in `src/lc2mesh/config.py`.

**`True STL unavailable ...; skipping shape metrics`** — expected for asteroids
4-10; their ground truth is not published. The lightcurve metrics are unaffected.

**`Reconstruction STL not found` from `visualize.py`** — run
`./run.sh <GPU_ID> <ASTEROID_ID>` first, or point `--recon-dir` at the directory
holding `reconstructed_ast<ID>.stl`.
