import numpy as np
import matplotlib.pyplot as plt


def plot_robot(transforms, ax=None):
    """
    Plot robot joint-frame origins as a 3D kinematic chain.

    Parameters
    ----------
    transforms : list of np.ndarray
        Base-to-frame transformations T00 ... T06.

    ax : matplotlib 3D axis, optional

    Returns
    -------
    ax : matplotlib 3D axis
    """

    positions = np.array([
        T[:3, 3] for T in transforms
    ])

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        marker="o",
        linewidth=3
    )

    # Frame numbers
    for i, p in enumerate(positions):
        ax.text(
            p[0],
            p[1],
            p[2],
            f"  {i}"
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title("UR5e Kinematic Chain")

        # Draw coordinate axes of each frame
    axis_length = 0.08

    for T in transforms:
        origin = T[:3, 3]
        R = T[:3, :3]

        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        ax.quiver(
            *origin, *x_axis,
            length=axis_length,
            normalize=True
        )

        ax.quiver(
            *origin, *y_axis,
            length=axis_length,
            normalize=True
        )

        ax.quiver(
            *origin, *z_axis,
            length=axis_length,
            normalize=True
        )

    # Equal axis scaling
    max_range = np.ptp(positions, axis=0).max() / 2.0
    midpoint = (
        positions.max(axis=0) +
        positions.min(axis=0)
    ) / 2.0

    ax.set_xlim(
        midpoint[0] - max_range,
        midpoint[0] + max_range
    )

    ax.set_ylim(
        midpoint[1] - max_range,
        midpoint[1] + max_range
    )

    ax.set_zlim(
        midpoint[2] - max_range,
        midpoint[2] + max_range
    )

    return ax