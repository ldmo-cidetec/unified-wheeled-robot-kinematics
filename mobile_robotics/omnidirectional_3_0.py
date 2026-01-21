from mobile_robotics.mobile_robot import MobileRobot
import numpy as np


class Omnidirectional30MobileRobot(MobileRobot):
    """
    Three-wheel omnidirectional platform (3,0) / Kiwi drive.

    Geometry:
    - Three omni wheels equally spaced by 120 degrees (2π/3) on a circle of radius L.
    - The body frame origin is at the geometric center of the wheel contact points.

    Kinematic structure:
    - Holonomic (rank(J) = 3).
    - J is constant and invertible for the symmetric configuration.

    Convention:
    - Wheel-space input u = [v1, v2, v3]^T are tangential wheel linear velocities (m/s).
    - Body twist xi_m = [x_dot_m, y_dot_m, omega]^T (m/s, m/s, rad/s).
    """

    def __init__(self, L: float, r: float):
        """
        Parameters
        ----------
        L : float
            Radial distance from the chassis center to each wheel contact point.
        r : float
            Wheel radius (assumed identical for all wheels).
        """
        super().__init__()

        if L <= 0:
            raise ValueError("Radius parameter L must be positive.")
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")

        self.L = float(L)
        self.r = float(r)

        # ------------------------------------------------------------------
        # Inverse kinematics: u = J * xi_m
        #
        # Each row corresponds to one wheel i and encodes its rolling direction u_i
        # and rotational contribution (c_i = L for the symmetric 3-wheel layout).
        # For the canonical (3,0) configuration, the Jacobian is constant.
        # ------------------------------------------------------------------
        self.J = np.array([
            [0.0,               1.0,  self.L],
            [-np.sqrt(3)/2.0,  -0.5,  self.L],
            [ np.sqrt(3)/2.0,  -0.5,  self.L],
        ], dtype=float)

        # ------------------------------------------------------------------
        # Forward kinematics: xi_m = J_inv * u
        #
        # For the symmetric (3,0) omni platform, J is square and non-singular,
        # hence J_inv = J^{-1}. The factor 1/3 yields the closed-form inverse.
        # ------------------------------------------------------------------
        self.J_inv = (1.0/3.0) * np.array([
            [0.0, -np.sqrt(3),  np.sqrt(3)],
            [2.0,        -1.0,        -1.0],
            [1.0/self.L,  1.0/self.L,  1.0/self.L],
        ], dtype=float)

    def __str__(self) -> str:
        return "omnidirectional_3_0"
