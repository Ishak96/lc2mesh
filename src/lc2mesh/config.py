"""Experiment configuration.

Every value here is taken verbatim from the reference notebook
(``inr_forward_train.ipynb``) so that ``run.py`` reproduces its results.
Only the asteroid id and the GPU id are supplied on the command line.
"""

from __future__ import annotations

CURVE_TYPE = "intensity"
SEED = 42
SPLIT_SEED = 1729
VALIDATION_FRACTION = 0.25
PRIOR_SHAPE_TYPE = "ellipsoid"
REFERENCE_MESH_SOURCE = "convex_inversion"
ALLOW_ORACLE_REFERENCE = False

N_SUBDIVISIONS = 4
INR_HIDDEN_DIM = 128
INR_NUM_LAYERS = 4
COARSE_W0 = 10.0  # low frequency: high w0 made the coarse radial field spiky (v*exp(field))
INR_FINAL_INIT_SCALE = 1e-3
GCN_ARCHITECTURE = "plain"
GCN_HIDDEN_DIM = 64 if GCN_ARCHITECTURE == "residual" else 128
GCN_NUM_LAYERS = 4 if GCN_ARCHITECTURE == "residual" else 6
GCN_FINAL_INIT_SCALE = 1e-3
LAMBERT_WEIGHT = 0.5

STAGE1_STEPS = 2000
LR_INR_STAGE1 = 1e-4
LAMBDA_CHAMFER = 0.3
HULL_N_SAMPLES = 4096
# Stage 1 lands the coarse INR on the convex-hull shape while fitting the
# lightcurves (convex forward model, no occlusion); the hull chamfer term is its
# geometric anchor. SELECTION_STAGE1="training" keeps converging toward the hull.
SELECTION_STAGE1 = "training"  # "training" -> keeps converging; "validation" -> early stop (~step 25)

# --- Non-convex (self-occlusion) forward model, Stage 2 only ---
# A convex forward model is blind to concavities, so no weight can make Stage 2
# carve non-convex features. Stage 2 multiplies a ray-cast SUN+OBSERVER
# visibility mask onto the convex matrix A: a concavity shadows its own facets,
# changing the lightcurve, so the GCN gets a gradient for WHERE to carve. Mask is
# recomputed every OCCLUSION_REFRESH_EVERY steps on a coarse OCCLUSION_N_PHASES
# rotation grid (nearest-upsampled to all phases).
OCCLUSION_ENABLED = True
OCCLUSION_REFRESH_EVERY = 100
OCCLUSION_N_PHASES = 24
OCCLUSION_EPS = 1e-6

STAGE2_STEPS = 2000
LR_GRAPH = 1e-3
LR_INR_STAGE2 = 1e-6
# Occlusion-driven Stage-2 deformation (INR frozen; only the GCN moves the mesh).
# Regularizers kept near-zero to let the GCN deform; a light one-ring Laplacian
# blocks pure-noise spikes. Deformation quality is limited by the GCN itself
# (plain graph convolutions over-smooth), so improving the refiner architecture
# matters more here than cranking LR / unfreezing the INR (that made it worse).
LAMBDA_CHAMFER_STAGE2 = 0.0
LAMBDA_EDGE_STAGE2 = 0.01
LAMBDA_CONTAINMENT_STAGE2 = 0.1
LAMBDA_LAPLACIAN = 0.1
LAMBDA_LAPLACIAN_STAGE2 = 0.2
LAMBDA_DISPLACEMENT_STAGE2 = 0.0
SELECTION_STAGE2 = "last"  # "last" -> keep final step | "training" -> best training obj | "validation" -> best held-out
# Convex-floor acceptance gate (safety net for convex bodies, e.g. the cube):
# the Stage-1 mesh's MSE under the CONVEX model is the floor. Stage 2 is kept
# only if its selected mesh, scored under occlusion physics with a FRESHLY
# ray-cast mask, beats that floor on the STAGE2_ACCEPTANCE cameras; otherwise
# the Stage-1 state is restored and the final mesh is labelled "stage1".
# "mean" gates on the average of the train and validation MSEs (equal weight
# per split, not per camera), a compromise between fit quality on the observed
# cameras and generalization to the held-out ones.
STAGE2_ACCEPTANCE = "mean"  # "mean" (avg of both) | "validation" (held-out) | "train"
EVAL_EVERY = 25
SCHEDULER_PATIENCE = 5

# Evaluation-only settings.
VOXEL_PITCH = 0.05  # official filled-voxel measure, pitch on cylinder-normalized meshes
CHAMFER_SURFACE_SAMPLES = 8192
CHAMFER_PITCH = 0.04
TARGET_CAM = "135_hor1"  # camera shown in the zoomed lightcurve panel

if STAGE2_ACCEPTANCE not in {"train", "validation", "mean"}:
    raise ValueError("STAGE2_ACCEPTANCE must be 'train', 'validation' or 'mean'")
