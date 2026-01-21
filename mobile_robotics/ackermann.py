from mobile_robotics.mobile_robot import MobileRobot
import numpy as np


class AckermannMobileRobot(MobileRobot):
    """
    Ackermann (car-like / bicycle) kinematic model with rear-wheel actuation.

    This class implements a simplified Ackermann steering model where:
    - The two rear wheels provide an equivalent longitudinal speed v = x_dot_m.
    - The steering angle delta defines the curvature via omega = (v / W) * tan(delta).

    Notes
    -----
    - The inverse-kinematics Jacobian (J) is identical to the differential-drive case
      because both rear wheels roll along the body x-axis.
    - The steering angle delta is incorporated in the forward kinematics (J_inv) by
      enforcing the bicycle-model relationship between omega and longitudinal speed.
    - If delta varies over time, update J_inv (or store delta and rebuild it) before
      calling direct_kinematics().
    """

    def __init__(self, L: float, W: float, r: float, delta: float):
        """
        Parameters
        ----------
        L : float
            Track width (distance between rear wheels).
        W : float
            Wheelbase (rear axle to front axle distance).
        r : float
            Wheel radius (assumed identical for the driven rear wheels).
        delta : float
            Steering angle (radians) of the equivalent front wheel (bicycle model).
        """
        super().__init__()

        if L <= 0:
            raise ValueError("Track width L must be positive.")
        if W <= 0:
            raise ValueError("Wheelbase W must be positive.")
        if r <= 0:
            raise ValueError("Wheel radius r must be positive.")

        self.r = r
        self.delta = float(delta)

        # ------------------------------------------------------------------
        # Inverse kinematics (rear wheel space)
        #
        # u = J * xi_m,   u = [v_left, v_right]^T
        # xi_m = [x_dot_m, y_dot_m, omega]^T
        #
        # Rear-wheel mapping matches the differential-drive structure.
        # ------------------------------------------------------------------
        self.J = np.array([
            [1.0, 0.0, -L / 2.0],
            [1.0, 0.0,  L / 2.0],
        ])

        # ------------------------------------------------------------------
        # Forward kinematics mapping (bicycle constraint embedded)
        #
        # x_dot_m = (v_left + v_right)/2
        # y_dot_m = 0
        # omega   = (x_dot_m / W) * tan(delta)
        #        = tan(delta)/(2W) * (v_left + v_right)
        # ------------------------------------------------------------------
        k = np.tan(self.delta) / (2.0 * W)
        self.J_inv = np.array([
            [0.5, 0.5],
            [0.0, 0.0],
            [k,   k  ],
        ])

    def set_steering(self, delta: float):
        """
        Update steering angle (delta) and rebuild the forward mapping.

        Use this method if delta is time-varying (e.g., during trajectory tracking).
        """
        self.delta = float(delta)

        # Rebuild only the row affected by delta
        # (requires W, which is embedded in the previous gain; store W if needed)
        # If you prefer, store W as self.W during __init__ for clarity.
        raise NotImplementedError(
            "If delta varies, store W as self.W and rebuild self.J_inv with k = tan(delta)/(2W)."
        )

    def __str__(self) -> str:
        return "ackermann"
