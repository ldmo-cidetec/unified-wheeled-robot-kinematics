from mobile_robotics.mobile_robot import MobileRobot
import numpy as np


class DifferentialMobileRobot(MobileRobot):
    """
    Differential-drive mobile robot model.

    This class implements the kinematic model of a standard differential-drive
    wheeled mobile robot using the unified Jacobian-based formulation.

    Geometry:
    - Two identical, independently actuated wheels.
    - Wheels are symmetrically located at ±L/2 along the lateral axis.
    - The body reference frame origin is located at the midpoint of the axle.

    Kinematic structure:
    - Non-holonomic (rank(J) = 2).
    - Forward and inverse kinematics are consistent with the classical
      unicycle/differential-drive model.
    """

    def __init__(self, L: float, r: float):
        """
        Parameters
        ----------
        L : float
            Distance between the left and right wheels (track width).
        r : float
            Radius of the wheels.
        """
        super().__init__()

        if L <= 0:
            raise ValueError("Track width L must be positive.")
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")

        self.r = r  # Wheel radius (assumed identical for both wheels)

        # ------------------------------------------------------------------
        # Inverse kinematics Jacobian (wheel space)
        #
        # u = J * xi_m
        #
        # u  : [v_left, v_right]^T
        # xi_m = [x_dot_m, y_dot_m, omega]^T
        #
        # y_dot_m = 0 is implicitly enforced by the rank deficiency.
        # ------------------------------------------------------------------
        self.J = np.array([
            [1.0, 0.0, -L / 2.0],
            [1.0, 0.0,  L / 2.0],
        ])

        # ------------------------------------------------------------------
        # Forward kinematics mapping (Moore–Penrose pseudo-inverse)
        #
        # xi_m = J_inv * u
        #
        # This mapping yields:
        #   x_dot_m = (v_left + v_right) / 2
        #   y_dot_m = 0
        #   omega   = (v_right - v_left) / L
        # ------------------------------------------------------------------
        self.J_inv = np.array([
            [ 0.5,   0.5 ],
            [ 0.0,   0.0 ],
            [-1.0/L, 1.0/L],
        ])

    def __str__(self) -> str:
        """Return a human-readable robot identifier."""
        return "differential"
