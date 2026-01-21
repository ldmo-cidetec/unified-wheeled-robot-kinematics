from abc import ABC, abstractmethod
import numpy as np


class MobileRobot(ABC):
    """
    Base class for wheeled mobile robots (WMRs) using a Jacobian-based kinematic model.

    Convention:
    - State: q = [x, y, theta]^T is expressed in the global frame.
    - Body twist: xi_m = [x_dot_m, y_dot_m, omega]^T is expressed in the body (mobile) frame.
    - Wheel-space input: u (e.g., tangential wheel linear velocities v_i).

    The mapping used is:
        xi = Rz(theta) * xi_m
        u  = J * xi_m
        xi_m = J_inv * u   (or pseudo-inverse, depending on the platform)
    """

    def __init__(self):
        super().__init__()
        self.r = None      # Wheel radius (scalar or array-like), used when converting v_i <-> omega_i
        self.J = None      # Inverse kinematics Jacobian: u = J * xi_m
        self.J_inv = None  # Forward kinematics mapping: xi_m = J_inv * u

    @staticmethod
    def _rotation_z(theta: float) -> np.ndarray:
        """2D planar rotation embedded in SE(2) twist coordinates."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]])

    def _check_ready(self):
        """Ensure model parameters are available before computing kinematics."""
        if self.J is None:
            raise ValueError("Jacobian J is not set. Define it in the concrete robot class.")
        if self.J_inv is None:
            raise ValueError("Mapping J_inv is not set. Define it in the concrete robot class.")
        # r is only required when converting between linear and angular wheel speeds

    # -------------------------------------------------------------------------
    # Forward/direct kinematics
    # -------------------------------------------------------------------------
    def direct_kinematics(self, q: np.ndarray, u: np.ndarray, angular_speed_inputs: bool = False) -> np.ndarray:
        """
        Compute time derivative q_dot = [x_dot, y_dot, theta_dot]^T in the global frame.

        Parameters
        ----------
        q : (3,) array_like
            Global pose [x, y, theta].
        u : (n,) array_like
            Wheel-space inputs. By default, u represents tangential wheel linear velocities v_i.
            If angular_speed_inputs=True, u is interpreted as wheel angular speeds omega_i (rad/s).
        angular_speed_inputs : bool
            If True, converts omega_i to v_i using v_i = r_i * omega_i.

        Returns
        -------
        q_dot : (3,) ndarray
            Global twist [x_dot, y_dot, omega].
        """
        self._check_ready()

        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 3:
            raise ValueError("q must be a 3-element vector: [x, y, theta].")

        u = np.asarray(u, dtype=float).reshape(-1)

        theta = q[2]
        R = self._rotation_z(theta)

        # Convert angular wheel speeds to tangential linear speeds, if requested.
        if angular_speed_inputs:
            if self.r is None:
                raise ValueError("Wheel radius r is required when angular_speed_inputs=True.")
            r = np.asarray(self.r, dtype=float)
            # Allow scalar r or per-wheel radii
            v = u * r if r.ndim == 0 else u * r.reshape(-1)
            q_body_dot = self.J_inv @ v
        else:
            q_body_dot = self.J_inv @ u

        # Body-frame twist -> global twist
        return R @ q_body_dot

    # -------------------------------------------------------------------------
    # Inverse kinematics
    # -------------------------------------------------------------------------
    def inverse_kinematics(self, q: np.ndarray, q_dot: np.ndarray, angular_speed_outputs: bool = False) -> np.ndarray:
        """
        Compute wheel-space commands u from pose q and desired global twist q_dot.

        Parameters
        ----------
        q : (3,) array_like
            Global pose [x, y, theta].
        q_dot : (3,) array_like
            Desired global twist [x_dot, y_dot, omega].
        angular_speed_outputs : bool
            If True, output wheel angular speeds omega_i (rad/s).
            Otherwise, output tangential wheel linear velocities v_i.

        Returns
        -------
        u : (n,) ndarray
            Wheel-space commands. v_i (default) or omega_i if angular_speed_outputs=True.
        """
        if self.J is None:
            raise ValueError("Jacobian J is not set. Define it in the concrete robot class.")

        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != 3:
            raise ValueError("q must be a 3-element vector: [x, y, theta].")

        q_dot = np.asarray(q_dot, dtype=float).reshape(-1)
        if q_dot.size != 3:
            raise ValueError("q_dot must be a 3-element vector: [x_dot, y_dot, omega].")

        theta = q[2]
        R = self._rotation_z(theta)

        # Global twist -> body twist
        q_body_dot = R.T @ q_dot

        # Body twist -> wheel-space
        v = self.J @ q_body_dot

        if angular_speed_outputs:
            if self.r is None:
                raise ValueError("Wheel radius r is required when angular_speed_outputs=True.")
            r = np.asarray(self.r, dtype=float)
            # omega_i = v_i / r_i
            return v / r if r.ndim == 0 else v / r.reshape(-1)

        return v

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable robot name/identifier."""
        raise NotImplementedError
