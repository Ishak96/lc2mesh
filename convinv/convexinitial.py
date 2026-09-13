"""Generate the convex-inversion prior for one asteroid.

Unchanged convex inversion (`ConvexInversionMulti`); only the CLI, the output
directory and the module imports were adapted to this repository. Results are
written to ``convex_inversions/`` with the filenames the INR pipeline expects:
``asteroid<ID>_<origin>_intensity.stl``.

Usage:
    python -m convinv.convexinitial <asteroid_number> <origin>
    python -m convinv.convexinitial 3 simulated
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from lc2mesh.constants import MODEL_BASE_RADII  # noqa: E402

from convinv.convex_inversion_multi import ConvexInversionMulti  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "convex_inversions"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convex-inversion prior generation for one asteroid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("asteroid_number", type=int, help="Asteroid id (1-10).")
    parser.add_argument(
        "origin", choices=("simulated", "real"), help="Lightcurve origin."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory the reconstruction STL and info file are written to.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing reconstruction instead of refusing to run.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    A = int(args.asteroid_number)  # asteroid number
    origin = args.origin  # e.g. simulated, observed, ...
    out_dir = Path(args.out_dir).resolve()

    if A not in MODEL_BASE_RADII:
        raise SystemExit(f"asteroid_number must be in 1-10, got {A}")

    # The shipped priors live here; never clobber them without --force.
    stl_path = out_dir / f"asteroid{A}_{origin}_intensity.stl"
    if stl_path.exists() and not args.force:
        print(
            f"{stl_path} already exists — refusing to overwrite.\n"
            "Pass --force to regenerate it, or --out-dir to write elsewhere."
        )
        return 0

    optimizer = "cg"

    CR = MODEL_BASE_RADII[A]
    CI = ConvexInversionMulti(cylinder_radius=CR, max_faces=1500)

    CI.load_lightcurve_data(
        model_num=A,
        origin=origin,
        curve_type="intensity",
    )

    LSLparam = 0.8
    CI.set_scattering_params(curve_type="intensity", value=LSLparam)

    CI.build_forward()
    print("total init areas", CI.areas_init_total)

    compute = True
    creg = 0.2

    while compute:
        print(f"Computing asteroid {A} with regularization parameter lambda={creg}", flush=True)

        CI.fit_areas(
            convexity_regularization=creg,
            optimizer=optimizer,
            scale_penalty=None,
            max_iter=30000,
            avoid_gpu=False,
        )

        print("area sum", np.sum(CI.areas_opt))

        CI.refine_areas()

        if CI.dark_area_percentage > 1:
            creg *= 2
        else:
            try:
                CI.minkowski_reconstruction(max_iter=20000, epsilon=0.005)
            except Exception as e:
                print(f"Error occurred while reconstructing asteroid {A}: {e}")
                compute = False
                continue

            v = CI.recon_vertices

            zmin, zmax = np.min(v[:, 2]), np.max(v[:, 2])
            v *= 2 / (zmax - zmin)

            minkowski_radius = np.max(np.linalg.norm(v[:, :2], axis=1))
            v *= CR / minkowski_radius

            zmin, zmax = np.min(v[:, 2]), np.max(v[:, 2])
            v[:, 2] -= zmin
            v[:, 2] *= 2 / (zmax - zmin)
            v[:, 2] -= 1

            CI.recon_vertices = v

            CI.save_reconstruction(out_dir=out_dir, views=False)

            with open(
                out_dir / f"asteroid{A}_{origin}_intensity_info.txt",
                "w",
            ) as f:
                f.write(f"Dark area percentage: {CI.dark_area_percentage}\n")
                f.write(f"Regularization parameter: {creg}\n")
                f.write(f"Area penalty parameter: {creg}\n")
                f.write(
                    f"Reconstruction radius (before cylinder normalization): "
                    f"{minkowski_radius}\n"
                )
                f.write(f"True radius: {CR}\n")
                f.write(f"LSL scattering parameter: {LSLparam}\n")
                f.write(f"Optimizer: {optimizer}\n")

            compute = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
