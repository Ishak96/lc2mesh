"""Official challenge voxel measures used for evaluation."""

import numpy as np
import trimesh
from scipy.ndimage import binary_fill_holes, label
from scipy.spatial.distance import cdist
from skimage.draw import polygon
from skimage.measure import find_contours


def calculate_voxel_measure(
    true_mesh: trimesh.Trimesh, pred_mesh: trimesh.Trimesh, pitch: float = 0.02
) -> float:
    """
    Fast voxel-based similarity measure.

    Parameters
    ----------
    true_mesh : trimesh.Trimesh
        Ground truth mesh.

    pred_mesh : trimesh.Trimesh
        Predicted mesh.

    pitch : float
        Voxel size. Smaller values give higher accuracy but require
        more memory and computation.

    Returns
    -------
    float
        Similarity score in [0, 1].
    """
    # Voxelize both meshes
    vox_true = true_mesh.voxelized(pitch)
    vox_pred = pred_mesh.voxelized(pitch)

    # Get occupied voxel indices
    true_voxels = set(map(tuple, vox_true.sparse_indices))
    pred_voxels = set(map(tuple, vox_pred.sparse_indices))

    if len(true_voxels) == 0 and len(pred_voxels) == 0:
        return 0.0

    # Symmetric difference
    mismatch_count = len(true_voxels.symmetric_difference(pred_voxels))

    # Denominator from your original formula
    total_volume_sum = len(true_voxels) + len(pred_voxels)

    score = 1.0 - mismatch_count / total_volume_sum

    return np.clip(score, 0.0, 1.0)


def load_as_trimesh(mshfile: str) -> trimesh.Trimesh:
    """Load a mesh file as a triangle `trimesh.Trimesh` (official grader loader).

    Uses ``meshio`` (matching the challenge implementation) to read the file and
    keep only triangle cells; falls back to ``trimesh.load_mesh`` when meshio is
    unavailable so common formats such as STL still load.
    """
    try:
        import meshio  # type: ignore[import-not-found]

        m = meshio.read(mshfile)
        return trimesh.Trimesh(vertices=m.points, faces=m.cells_dict["triangle"])
    except ImportError:
        return trimesh.load_mesh(mshfile)


def _as_trimesh(mesh: trimesh.Trimesh | str) -> trimesh.Trimesh:
    return mesh if isinstance(mesh, trimesh.Trimesh) else load_as_trimesh(mesh)


def relative_volume_difference_voxelized(
    mesh1: trimesh.Trimesh | str,
    mesh2: trimesh.Trimesh | str,
    pitch: float = 0.05,
) -> tuple[float, float]:
    """Official challenge voxel measures on a shared, aligned voxel grid.

    Both meshes are padded with the SAME two ghost boxes (at the shared
    bounding-box corners) so their filled voxel grids share an origin, pitch,
    and shape, making an element-wise comparison valid. This is what makes the
    measure translation-sensitive; comparing each mesh's own voxel indices would
    silently ignore a global offset.

    Returns ``(measure1, measure2)`` where both are DIFFERENCES (0 = identical,
    larger = more different):

    - ``measure1 = 1 - |A ∩ B| / |A ∪ B|`` (one minus voxel IoU)
    - ``measure2 = (|A \\ B| + |B \\ A|) / (|A| + |B|)`` (symmetric difference)
    """
    A = _as_trimesh(mesh1)
    B = _as_trimesh(mesh2)

    min_bound = np.minimum(A.bounds[0], B.bounds[0])
    max_bound = np.maximum(A.bounds[1], B.bounds[1])

    ghost_min = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    ghost_min.apply_translation(min_bound)
    ghost_max = trimesh.creation.box(extents=np.array([pitch, pitch, pitch]))
    ghost_max.apply_translation(max_bound)

    A_padded = trimesh.util.concatenate([A, ghost_min, ghost_max])
    B_padded = trimesh.util.concatenate([B, ghost_min, ghost_max])

    filled_A = A_padded.voxelized(pitch, method="subdivide").fill().matrix.astype(bool)
    filled_B = B_padded.voxelized(pitch, method="subdivide").fill().matrix.astype(bool)

    # Shared origin and pitch, so any float-rounding shape mismatch is a trailing
    # off-by-one; embed both grids at the common origin to stay aligned.
    shape = np.maximum(filled_A.shape, filled_B.shape)

    def embed(matrix: np.ndarray) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        out[: matrix.shape[0], : matrix.shape[1], : matrix.shape[2]] = matrix
        return out

    filled_A = embed(filled_A)
    filled_B = embed(filled_B)

    intersection = np.logical_and(filled_A, filled_B).sum()
    union = np.logical_or(filled_A, filled_B).sum()
    diff_ab = np.logical_and(filled_A, np.logical_not(filled_B)).sum()
    diff_ba = np.logical_and(filled_B, np.logical_not(filled_A)).sum()
    vol_a = int(filled_A.sum())
    vol_b = int(filled_B.sum())

    measure1 = 1.0 - intersection / union if union else 0.0
    measure2 = (diff_ab + diff_ba) / (vol_a + vol_b) if (vol_a + vol_b) else 0.0
    return float(measure1), float(measure2)


def _resample_boundary(curve, n):
    d = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    s = np.concatenate([[0], np.cumsum(d)])

    if s[-1] == 0:
        return np.repeat(curve[:1], n, axis=0)

    s_new = np.linspace(0, s[-1], n)

    return np.column_stack([
        np.interp(s_new, s, curve[:, 0]),
        np.interp(s_new, s, curve[:, 1]),
    ])


def _largest_component(mask):
    labeled, n = label(mask, structure=np.ones((3, 3)))

    if n == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    return labeled == np.argmax(sizes)


def _rasterize(vertices, faces, img_size, xmin, ymin, grid_range):
    x = (vertices[:, 0] - xmin) / grid_range * (img_size - 1)
    y = (vertices[:, 1] - ymin) / grid_range * (img_size - 1)

    mask = np.zeros((img_size, img_size), dtype=bool)

    for face in faces:
        rr, cc = polygon(y[face], x[face], shape=mask.shape)
        mask[rr, cc] = True

    mask = binary_fill_holes(mask)
    return _largest_component(mask)


def calculate_2d_metric(
    meshT: trimesh.Trimesh | None,
    meshRec: trimesh.Trimesh | None,
    theta: float,
    img_size: int = 500,
    n_boundary: int = 1000,
) -> float:
    """Official challenge score calculation for 2D mesh projection.
    Args:
        meshT (trimesh.Trimesh | None): Ground truth mesh.
        meshRec (trimesh.Trimesh | None): Reconstructed mesh.
        theta (float): Rotation angle in degrees.
        img_size (int, optional): Size of the rasterized image. Defaults to 500.
        n_boundary (int, optional): Number of points to resample the boundary. Defaults to 1000.

    Returns:
        float: 2D mesh projection metric for the given rotation angle theta.
    """
    gt, recon = _as_trimesh(meshT), _as_trimesh(meshRec)
    V_gt, F_gt = np.asarray(gt.vertices, dtype=float), np.asarray(gt.faces)
    V_recon, F_recon = np.asarray(recon.vertices, dtype=float), np.asarray(recon.faces)

    V_gt -= V_gt.mean(axis=0)
    V_recon -= V_recon.mean(axis=0)

    theta = np.deg2rad(theta)

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])

    proj_gt, proj_recon = (V_gt @ R.T)[:, :2], (V_recon @ R.T)[:, :2]

    xmin, ymin = proj_gt.min(axis=0)
    xmax, ymax = proj_gt.max(axis=0)

    gt_diag = np.hypot(xmax - xmin, ymax - ymin)

    margin = 0.05 * gt_diag
    xmin -= margin; xmax += margin
    ymin -= margin; ymax += margin

    grid_range = max(xmax - xmin, ymax - ymin)
    BW_gt = _rasterize(proj_gt, F_gt, img_size, xmin, ymin, grid_range)
    BW_recon = _rasterize(proj_recon, F_recon, img_size, xmin, ymin, grid_range)

    gt_contours = find_contours(BW_gt.astype(float), 0.5)
    recon_contours = find_contours(BW_recon.astype(float), 0.5)

    if not gt_contours or not recon_contours:
        return 0.0

    gt_boundary = max(gt_contours, key=len)[:, ::-1]
    recon_boundary = max(recon_contours, key=len)[:, ::-1]
    gt_boundary = _resample_boundary(gt_boundary, n_boundary)
    recon_boundary = _resample_boundary(recon_boundary, n_boundary)

    gt_diag_px = gt_diag / grid_range * (img_size - 1)

    D = cdist(recon_boundary, gt_boundary)
    d1, d2 = D.min(axis=1), D.min(axis=0)

    rms = np.sqrt(np.mean(np.concatenate([d1**2, d2**2])))

    return float(max(0.0, 1.0 - rms / gt_diag_px))