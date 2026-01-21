from mobile_robotics.differential import DifferentialMobileRobot
from mobile_robotics.ackermann import AckermannMobileRobot
from mobile_robotics.omnidirectional_3_0 import Omnidirectional30MobileRobot
from mobile_robotics.mecanum import MecanumMobileRobot

import numpy as np
from matplotlib import pyplot as plt


def plot_trajectory(x, y, theta, t, square_margin_ratio=0.05):
    """
    Plot planar trajectory and orientation over time.

    Parameters
    ----------
    x, y : (n,) array_like
        Global position coordinates.
    theta : (n,) array_like
        Global heading angle (rad).
    t : (n,) array_like
        Time vector (s).
    square_margin_ratio : float
        Relative margin added to the square plot limits (e.g., 0.05 = 5%).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    if not (x.size == y.size == theta.size == t.size):
        raise ValueError("x, y, theta, and t must have the same length.")

    # ---- Square axis limits based on the largest trajectory dimension ----
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    L = max(x_range, y_range)
    if L == 0.0:
        L = 1.0  # degenerate case: robot does not move

    x_c = 0.5 * (x_max + x_min)
    y_c = 0.5 * (y_max + y_min)
    margin = square_margin_ratio * L

    # -------------------------
    # Figure 1: Planar motion
    # -------------------------
    fig1 = plt.figure(figsize=(6, 6))
    ax1 = fig1.gca()
    ax1.set_title("Robot position")
    ax1.plot(x, y, "k-", linewidth=2, label="Trajectory")
    ax1.scatter(x[0], y[0], color="g", s=40, zorder=3, label="Initial point")
    ax1.scatter(x[-1], y[-1], color="r", s=40, zorder=3, label="Final point")

    ax1.set_xlim(x_c - L / 2 - margin, x_c + L / 2 + margin)
    ax1.set_ylim(y_c - L / 2 - margin, y_c + L / 2 + margin)

    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True)
    ax1.set_xlabel(r"$x$ (m)")
    ax1.set_ylabel(r"$y$ (m)")
    ax1.legend(loc="best")
    fig1.tight_layout()

    # -------------------------
    # Figure 2: Orientation
    # -------------------------
    fig2 = plt.figure(figsize=(7, 3.5))
    ax2 = fig2.gca()
    ax2.set_title("Robot orientation")
    ax2.plot(t, theta, "k-", linewidth=2)

    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True)
    ax2.set_xlabel(r"$t$ (s)")
    ax2.set_ylabel(r"$\theta$ (rad)")
    fig2.tight_layout()

    plt.show()


def simulate_kinematics(robot, q_0, t_f, dt, u):
    """
    Open-loop kinematic simulation using forward (direct) kinematics.

    The robot state evolves as:
        q_{k+1} = q_k + q_dot(q_k, u_k) * dt
    where q = [x, y, theta]^T is expressed in the global frame.

    Parameters
    ----------
    robot : MobileRobot
        Robot instance providing direct_kinematics().
    q_0 : (3,) array_like
        Initial state [x0, y0, theta0].
    t_f : float
        Final time (s).
    dt : float
        Sampling period (s).
    u : (n, m) array_like
        Wheel commands over time. m must match the number of actuated wheels
        assumed by the robot model (rows/cols compatible with J_inv).
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_f <= 0:
        raise ValueError("t_f must be positive.")

    q_0 = np.asarray(q_0, dtype=float).reshape(-1)
    if q_0.size != 3:
        raise ValueError("q_0 must be a 3-element vector: [x0, y0, theta0].")

    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError("u must be a 2D array with shape (n_steps, n_inputs).")

    # Number of steps consistent with dt and t_f
    n = int(t_f / dt) + 1
    if u.shape[0] != n:
        raise ValueError(f"u must have {n} rows (one per time step). Got {u.shape[0]}.")

    # Optional: basic compatibility check with robot forward mapping
    if getattr(robot, "J_inv", None) is None:
        raise ValueError("robot.J_inv is not defined; cannot simulate forward kinematics.")
    if robot.J_inv.shape[1] != u.shape[1]:
        raise ValueError(
            f"Input dimension mismatch: robot expects {robot.J_inv.shape[1]} wheel inputs, "
            f"but u has {u.shape[1]} columns."
        )

    q = np.zeros((n, 3), dtype=float)
    q[0, :] = q_0

    # Time vector
    t = np.linspace(0.0, t_f, n)

    # Kinematic simulation (explicit Euler)
    for k in range(n - 1):
        q_dot = robot.direct_kinematics(q=q[k, :], u=u[k, :])
        q[k + 1, :] = q[k, :] + q_dot * dt

    plot_trajectory(x=q[:, 0], y=q[:, 1], theta=q[:, 2], t=t)


def main():
    """
    Example simulations for different wheeled mobile robot configurations.

    Each case uses constant wheel commands in open-loop to illustrate
    the characteristic kinematic behavior of each platform.
    """

    # -------------------------
    # Global simulation config
    # -------------------------
    t_f = 10.0            # Total simulation time [s]
    dt = 0.01             # Sampling period [s]
    n = int(t_f / dt) + 1 # Number of simulation steps
    q_0 = np.array([0.0, 0.0, 0.0])  # Initial pose [x, y, theta]

    # ============================================================
    # Differential drive: circular trajectory
    # ============================================================
    diff = DifferentialMobileRobot(
        L=0.30,   # Distance between left and right wheels [m]
        r=0.05    # Wheel radius [m]
    )

    # Wheel linear velocities [m/s]
    v_left  = 0.08
    v_right = 0.12

    # For differential drive:
    #   v = (v_left + v_right)/2
    #   omega = (v_right - v_left)/L
    #
    # Since both v and omega are constant and nonzero,
    # the robot follows a circular trajectory with constant curvature.
    u_diff = np.zeros((n, 2), dtype=float)
    u_diff[:, 0] = v_left   # Left wheel velocity
    u_diff[:, 1] = v_right  # Right wheel velocity

    simulate_kinematics(
        robot=diff,
        q_0=q_0,
        t_f=t_f,
        dt=dt,
        u=u_diff
    )

    # ============================================================
    # Ackermann (bicycle model): circular trajectory
    # ============================================================
    ack = AckermannMobileRobot(
        L=0.30,                 # Rear track width [m]
        W=0.50,                 # Wheelbase (rear axle to front axle) [m]
        r=0.05,                 # Rear wheel radius [m]
        delta=np.deg2rad(20.0)  # Steering angle [rad]
    )

    # Rear wheel linear velocity [m/s]
    v_rear = 0.10

    # In the Ackermann model:
    #   x_dot_m = v_rear
    #   omega = (x_dot_m / W) * tan(delta)
    #
    # With constant v_rear and constant delta:
    # - x_dot_m is constant
    # - omega is constant
    #
    # Therefore, the vehicle follows a circular trajectory
    # with radius R = W / tan(delta).
    u_ack = np.zeros((n, 2), dtype=float)
    u_ack[:, 0] = v_rear  # Left rear wheel
    u_ack[:, 1] = v_rear  # Right rear wheel

    simulate_kinematics(
        robot=ack,
        q_0=q_0,
        t_f=t_f,
        dt=dt,
        u=u_ack
    )

    # ============================================================
    # Omnidirectional (3,0): forward translation
    # ============================================================
    omni = Omnidirectional30MobileRobot(
        L=0.25,  # Distance from center to each wheel [m]
        r=0.05   # Wheel radius [m]
    )

    # Desired global velocity [x_dot, y_dot, omega]
    # x_dot > 0, y_dot = 0, omega = 0
    #
    # Expected result:
    # - Pure forward translation along the x-axis
    # - No lateral motion
    # - No change in orientation
    q_dot_desired = np.array([0.12, 0.0, 0.0])

    # Compute wheel velocities that realize the desired motion
    u_single = omni.inverse_kinematics(
        q=q_0,
        q_dot=q_dot_desired,
        angular_speed_outputs=False
    )

    u_omni = np.zeros((n, 3), dtype=float)
    u_omni[:, :] = u_single  # Same wheel commands at all time steps

    simulate_kinematics(
        robot=omni,
        q_0=q_0,
        t_f=t_f,
        dt=dt,
        u=u_omni
    )

    # ============================================================
    # Mecanum 4WD: diagonal translation without rotation
    # ============================================================
    mec = MecanumMobileRobot(
        L=0.20,  # Half-length of the chassis [m]
        W=0.20,  # Half-width of the chassis [m]
        r=0.05   # Wheel radius [m]
    )

    # Wheel linear velocities [m/s]
    #
    # For the chosen Jacobian structure:
    #   v1 = x_dot_m - y_dot_m - (L+W)*omega
    #   v4 = x_dot_m - y_dot_m + (L+W)*omega
    #
    # Setting v1 = v4 and v2 = v3 = 0 yields:
    # - omega = 0
    # - x_dot_m = y_dot_m != 0
    #
    # Expected result:
    # - Pure diagonal translation
    # - No change in orientation
    u_mec = np.zeros((n, 4), dtype=float)
    u_mec[:, 0] = 0.10  # Wheel 1
    u_mec[:, 3] = 0.10  # Wheel 4

    simulate_kinematics(
        robot=mec,
        q_0=q_0,
        t_f=t_f,
        dt=dt,
        u=u_mec
    )



if __name__ == "__main__":
    main()
