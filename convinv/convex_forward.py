"""Observer/sun unit vectors for the challenge camera geometry.

Extracted verbatim from the original `hac.convex_forward`.
"""

import numpy as np


def get_unit_vectors(phi_deg, alpha_deg):
    """
    Compute the unit vectors for the light source and camera.

    Parameters:
    - phi_deg: Azimuthal angle (degrees) between light source and camera.
    - alpha_deg: Elevation angle (degrees) between xy-plane and camera.

    Returns:
    - light_vector: Unit vector for the light source (fixed at (-1, 0, 0)).
    - camera_vector: Unit vector for the camera.
    """
    # Convert angles to radians
    phi = np.radians(phi_deg - 180)  # Adjust phi to align with negative x-axis
    alpha = np.radians(alpha_deg)

    # Light source unit vector (fixed)
    light_vector = np.array([-1, 0, 0])

    # Camera unit vector
    camera_vector = np.array(
        [np.cos(phi) * np.cos(alpha), np.sin(phi) * np.cos(alpha), np.sin(alpha)]
    )

    return light_vector, camera_vector
