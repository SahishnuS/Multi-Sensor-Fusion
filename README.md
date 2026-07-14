# Multi-Sensor Fusion for Underwater Navigation — Error-State EKF (AUV)

A navigation state-estimation framework for a survey-class Autonomous Underwater Vehicle (AUV), fusing **IMU, pressure (depth), DVL, GPS (surface-only), magnetometer, and visual odometry** through an **Error-State Extended Kalman Filter (ES-EKF)**.

No recorded underwater sensor dataset was available for this project, so a realistic multi-sensor dataset is generated from physics-based vehicle motion with configurable Gaussian noise, IMU bias drift, DVL dropouts, and intermittent GPS availability — emulating real AUV operating conditions.

## Results

| Metric | Result |
|---|---|
| Position RMSE | 1.39 m (hundreds of meters, multi-minute GPS-denied dives) |
| Velocity RMSE | 0.071 m/s |
| Yaw RMSE | 0.71° |
| Final position error at mission end | 0.35 m |
| Accelerometer bias (estimated vs. true) | converges to within ~0.01 m/s² |
| Gyroscope bias (estimated vs. true) | converges to within ~0.001 rad/s |

## Why Error-State EKF

| Approach | Computational Cost | Nonlinear Handling | Embedded Suitability |
|---|---|---|---|
| Classical (linear) KF | Lowest | Invalid for attitude/nonlinear dynamics | Not applicable |
| Standard EKF (quaternion in state) | Moderate | First-order, but singular/over-parameterized covariance | Workable but messy |
| **Error-State EKF (this project)** | Moderate | First-order on a minimal 3-parameter attitude error | **Excellent — flight-proven in production AUV/AHRS/INS stacks** |
| Unscented KF (UKF) | High | Second-order, no explicit Jacobians | Heavier for real-time embedded loops |
| Particle Filter | Very high | Handles arbitrary nonlinearity | Impractical at IMU rate |

ES-EKF was chosen for its small-attitude-error linearization accuracy, efficient 3-parameter rotation-vector attitude representation (no quaternion covariance singularity), continuous online bias estimation, low computational cost (15-state error model), and status as the de facto industry standard for INS/DVL/GNSS navigation.

## Sensor Suite

| Sensor | Rate | Measures | Noise (1σ) | Failure Mode |
|---|---|---|---|---|
| IMU (accel + gyro) | 100 Hz | Specific force, angular rate | 0.02 m/s²·√Hz accel, 0.001 rad/s·√Hz gyro | Saturation, unbounded drift if unaided |
| Depth (pressure) | 10 Hz | Vertical position (z) | 0.03 m | No horizontal info |
| DVL | 7 Hz | Body-frame velocity | 0.02 m/s | Bottom-lock loss (modeled: 1%/epoch dropout) |
| GPS | 1 Hz | Horizontal position | 2.5 m | Surface-only, gated on "surfaced" condition |
| Magnetometer | 20 Hz | Body-frame magnetic field | 0.03 (unit-normalized) | Hard/soft-iron distortion |
| Camera / Visual Odometry | 10 Hz | Body-frame velocity (relative) | 0.06 m/s | Drifts over time; dropouts (modeled: 15%/epoch) |

## Fusion Architecture

1. **IMU Prediction** — propagates position, velocity, orientation, and covariance at 100 Hz.
2. **Depth Update** — corrects vertical position.
3. **DVL Update** — corrects velocity via body-frame bottom-track return.
4. **GPS Update** — absolute position correction, gated to surfaced windows only.
5. **Magnetometer Update** — corrects heading against Earth's magnetic field.
6. **Visual Odometry Update** — optional secondary velocity source during DVL outages.
7. **Correction Step** — innovation + Kalman gain → error-state injection → covariance update (Joseph form).

The filter carries a 15-element error state: position (3), velocity (3), attitude error (3), accelerometer bias (3), gyroscope bias (3).

## Repository Contents

```
.
├── es_ekf_auv.py              # Full ES-EKF implementation: quaternion math,
│                               #   mission/sensor simulation, filter, driver
├── generate_plots.py           # Generates the full plot suite from saved output
└── plots/                      # 8-plot visualization suite (generated)
    ├── 01_trajectory_2d.png
    ├── 02_trajectory_3d.png
    ├── 03_position_est_vs_truth.png
    ├── 04_velocity_components.png
    ├── 05_orientation_rpy.png
    ├── 06_position_error.png
    ├── 07_velocity_error.png
    ├── 08_orientation_error.png
```

`es_ekf_auv.py` is organized into six modules: quaternion utilities, a `SensorConfig` dataclass (every noise/bias/rate/dropout parameter in one place), a 600 s lawnmower-survey ground-truth generator, a sensor simulator, the `ESEKF` class (`predict()` + one `update_*()` per sensor), and a simulation driver (`run_filter`).

## Requirements

```
python >= 3.9
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

## Usage

Run the mission simulation and filter:

```bash
python3 es_ekf_auv.py
```

This generates the 600 s ground-truth trajectory, simulates all sensor streams, runs the ES-EKF, prints position/velocity RMSE to the console, and saves `sim_output.npz` and `sim_extra.pkl`.

Generate the plot suite from the saved output:

```bash
python3 generate_plots.py
```

This reads `sim_output.npz` / `sim_extra.pkl` and writes all 8 plots to `plots/`.

## Using Your Own Sensor Data

The simulator is fully parameterized through the `SensorConfig` dataclass, so a recorded AUV dataset (e.g. from a MOOS-IvP, LCM, or ROS2 bag) can be substituted by replacing the `generate_ground_truth` / `simulate_sensors` calls with a data loader that produces the same timestamped measurement arrays — no changes to the filter logic (`ESEKF`) are required.

## Limitations

- First-order linearization can degrade under very aggressive, high-angular-rate maneuvers.
- Single Gaussian-noise-per-sensor model does not capture heavy-tailed outliers (e.g. DVL multipath) — no innovation-based/chi-squared gating implemented.
- Accelerometer/gyro biases are modeled as pure random walks, a simplification of real turn-on-to-turn-on bias behavior.
- Euler (first-order) process integration rather than a higher-order integrator.
- Reset Jacobian on error-state injection is simplified to identity rather than the exact (I − ½[δθ]ₓ) form.
- Sensor data is simulated, not field-recorded.

## Future Improvements

- Visual-Inertial Navigation (VIO/VINS-style tight coupling)
- Sonar-inertial fusion for terrain-relative navigation
- Adaptive Kalman filtering (online noise-covariance estimation)
- Factor graph optimization (GTSAM) for offline reprocessing / loop closure
- ROS2 integration (`ESEKF` as a node, `sensor_msgs` in / `nav_msgs/Odometry` out)
- NVIDIA Jetson deployment with a compiled/embedded linear-algebra path
- SLAM integration for bounded-error navigation without surfacing
- AI-based sensor fault detection (DVL multipath, magnetometer distortion, VO tracking failure)

## Author

Sahishnu S
