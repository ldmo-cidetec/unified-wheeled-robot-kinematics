# Wheeled Mobile Robot Kinematics Library

This repository provides a lightweight and self-contained **Python library for the kinematic modeling and simulation of wheeled mobile robots** using a unified Jacobian-based formulation.  
The implementation corresponds to and supports the results presented in the paper:

**“Generalized Kinematic Modeling of Wheeled Mobile Robots: A Unified Framework for Heterogeneous Architectures.”**

The code focuses on:
- Forward (direct) and inverse kinematics
- Open-loop simulation in the plane
- Clear separation between robot models and simulation scripts
- Readable, extensible, and educational code

The library is intended for:
- Research prototyping
- Teaching mobile robotics and kinematics
- Reproducible simulation examples

---

## Supported Robot Models

The following wheeled mobile robot configurations are included:

- **Differential-drive robot**
- **Ackermann (car-like / bicycle) robot**
- **Omnidirectional robot (3,0 configuration)**
- **Mecanum 4WD robot**

All models are implemented using a common base class and follow consistent conventions.

---

## Project Structure

```
Project/
├── mobile_robotics/
│   ├── __init__.py
│   ├── mobile_robot.py
│   ├── differential.py
│   ├── ackermann.py
│   ├── omnidirectional_3_0.py
│   └── mecanum.py
│
└── kinematics_simulations.py
```

- **mobile_robotics/**  
  Contains the robot models and the abstract base class.

- **kinematics_simulations.py**  
  Main script with example simulations for each robot type.

---

## Kinematic Conventions

All robot models share the same state representation:

- **Pose**:  
  $q = [x, y, θ]^T$ expressed in the global frame.

- **Body-frame twist**:  
  $\xi_m = [\dot{x}_m, \dot{y}_m, \omega]^T$

The relationship between body-frame and global velocities follows standard planar rigid-body kinematics.

Wheel commands are expressed as **tangential wheel linear velocities** (m/s) by default.

---

## Running the Examples

From the project root directory, run:

```
python kinematics_simulations.py
```

The script executes a sequence of open-loop simulations illustrating typical motions:

- Circular motion for differential-drive robots  
- Circular motion for Ackermann robots with constant steering  
- Pure translation for omnidirectional (3,0) robots  
- Diagonal translation for mecanum robots  

For each case, the following plots are generated:
- Planar trajectory: $(x, y)$
- Orientation: $\theta(t)$

---

## Adding a New Robot Model

To add a new wheeled mobile robot configuration:

### 1. Create a New Robot Class

Add a new file in `mobile_robotics/`, for example:

```
mobile_robotics/my_new_robot.py
```

The class should:
- Inherit from `MobileRobot`
- Define the matrices:
  - `J` (inverse kinematics mapping)
  - `J_inv` (forward kinematics mapping)
- Define geometric parameters (wheel radius, offsets, etc.)

Use existing robots as references:
- `differential.py`
- `ackermann.py`
- `omnidirectional_3_0.py`
- `mecanum.py`

---

## Numerical Integration

All simulations use **explicit Euler integration**:

$q_{k+1}$ = $q_k+\dot{q}_k\,\Delta t$

This choice is intentional:
- Simple and transparent
- Sufficient for kinematic validation
- Easy to replace if higher-order integration is required

---

## Dependencies

The project uses only standard scientific Python libraries:

- Python ≥ 3.8
- NumPy
- Matplotlib

No additional frameworks or build tools are required.

---

## Design Principles

- Minimal dependencies  
- Clear mathematical correspondence  
- Explicit assumptions and conventions  
- Easy extensibility  
- Suitable for education and research  

