from mobile_robotics.mobile_robot import MobileRobot
import numpy as np


class MecanumMobileRobot(MobileRobot):
    """
    Four-wheel mecanum (4WD) holonomic mobile robot.

    Geometry:
    - Four mecanum wheels located at the corners of a rectangle.
    - The body frame origin is at the geometric center of the wheel contact points.
    - L: half-length (longitudinal offset to wheel contact points)
    - W: half-width  (lateral offset to wheel contact points)

    Kinematic structure:
    - Holonomic (rank(J) = 3) with redundancy (4 wheels for 3 DoF twist).
    - Uses the standard mecanum mapping with the (L + W) coupling term.

    Convention:
    - Wheel-space input u = [v1, v2, v3, v4]^T are tangential wheel linear velocities (m/s).
    - Body twist xi_m = [x_dot_m, y_dot_m, omega]^T.
    - IMPORTANT: The signs in J depend on the wheel indexing and roller orientation.
      Here we assume a fixed wheel ordering consistent with the provided matrix.
      If your wheel numbering differs, permute the rows of J accordingly.
    """

    def __init__(self, L: float, W: float, r: float):
        """
        Parameters
        ----------
        L : float
            Half-length of the chassis (front/rear offset from center to wheel contact points).
        W : float
            Half-width of the chassis (left/right offset from center to wheel contact points).
        r : float
            Wheel radius (assumed identical for all wheels).
        """
        super().__init__()

        if L <= 0:
            raise ValueError("Half-length L must be positive.")
        if W <= 0:
            raise ValueError("Half-width W must be positive.")
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")

        self.L = float(L)
        self.W = float(W)
        self.r = float(r)

        k = self.L + self.W  # Standard mecanum coupling term

        # ------------------------------------------------------------------
        # Inverse kinematics: u = J * xi_m
        #
        # For the chosen wheel convention, the mapping is:
        #   v1 = x_dot_m - y_dot_m - k * omega
        #   v2 = x_dot_m + y_dot_m + k * omega
        #   v3 = x_dot_m + y_dot_m - k * omega
        #   v4 = x_dot_m - y_dot_m + k * omega
        #
        # This matches the canonical mecanum 4WD model.
        # ------------------------------------------------------------------
        self.J = np.array([
            [1.0, -1.0, -k],
            [1.0,  1.0,  k],
            [1.0,  1.0, -k],
            [1.0, -1.0,  k],
        ], dtype=float)

        # ------------------------------------------------------------------
        # Forward kinematics: xi_m = J_inv * u
        #
        # Redundant holonomic system -> use a closed-form pseudo-inverse
        # for this symmetric mecanum structure:
        #   x_dot_m = (v1 + v2 + v3 + v4) / 4
        #   y_dot_m = (-v1 + v2 + v3 - v4) / 4
        #   omega   = (-v1 + v2 - v3 + v4) / (4k)
        # ------------------------------------------------------------------
        self.J_inv = 0.25 * np.array([
            [1.0,      1.0,      1.0,      1.0],
            [-1.0,     1.0,      1.0,     -1.0],
            [-1.0/k,   1.0/k,   -1.0/k,    1.0/k],
        ], dtype=float)

    def __str__(self) -> str:
        return "mecanum"
