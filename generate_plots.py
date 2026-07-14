"""
Generates the full publication-quality plot suite for the ES-EKF AUV
navigation results: trajectories, state comparisons, errors, RMSE,
innovations, and covariance evolution.
"""
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from es_ekf_auv import quat_to_euler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

d = np.load(os.path.join(HERE, "sim_output.npz"))
with open(os.path.join(HERE, "sim_extra.pkl"), "rb") as fh:
    extra = pickle.load(fh)

t = d["t"]
pos, vel, quat = d["pos"], d["vel"], d["quat"]
est_pos, est_vel, est_eul = d["est_pos"], d["est_vel"], d["est_eul"]
est_ba, est_bg, P_diag = d["est_ba"], d["est_bg"], d["P_diag"]
surface_windows = d["surface_windows"]

eul_true = np.array([quat_to_euler(q) for q in quat])

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.grid": True,
    "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})

def shade_surface(ax):
    for (a, b) in surface_windows:
        ax.axvspan(a, b, color="gold", alpha=0.15, label="_nolegend_")

def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------- 1. 2D traj
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(pos[:, 1], pos[:, 0], label="Ground Truth", color="black", lw=2, alpha=0.8)
ax.plot(est_pos[:, 1], est_pos[:, 0], label="ES-EKF Estimate", color="crimson", lw=1.2, ls="--")
ax.scatter(pos[0, 1], pos[0, 0], c="green", marker="o", s=80, zorder=5, label="Start")
ax.scatter(pos[-1, 1], pos[-1, 0], c="blue", marker="X", s=80, zorder=5, label="End")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)")
ax.set_title("2D Vehicle Trajectory (Lawnmower Survey)")
ax.legend(); ax.set_aspect("equal", adjustable="datalim")
savefig("01_trajectory_2d")

# ---------------------------------------------------------------- 2. 3D traj
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(pos[:, 1], pos[:, 0], -pos[:, 2], label="Ground Truth", color="black", lw=2)
ax.plot(est_pos[:, 1], est_pos[:, 0], -est_pos[:, 2], label="ES-EKF Estimate", color="crimson", lw=1, ls="--")
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Altitude (m, +up)")
ax.set_title("3D Vehicle Trajectory")
ax.legend()
savefig("02_trajectory_3d")

# ---------------------------------------------------------------- 3. Est vs GT position (xyz vs t)
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
labels = ["North (m)", "East (m)", "Depth (m)"]
for i in range(3):
    axs[i].plot(t, pos[:, i], color="black", lw=1.6, label="Ground Truth")
    axs[i].plot(t, est_pos[:, i], color="crimson", lw=1, ls="--", label="ES-EKF Estimate")
    shade_surface(axs[i])
    axs[i].set_ylabel(labels[i])
axs[0].legend(loc="upper right"); axs[0].set_title("Estimated vs Ground-Truth Position (gold = surfaced/GPS)")
axs[2].set_xlabel("Time (s)")
savefig("03_position_est_vs_truth")

# ---------------------------------------------------------------- 4. Velocity components
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
vlabels = ["$v_x$ North (m/s)", "$v_y$ East (m/s)", "$v_z$ Down (m/s)"]
for i in range(3):
    axs[i].plot(t, vel[:, i], color="black", lw=1.6, label="Ground Truth")
    axs[i].plot(t, est_vel[:, i], color="steelblue", lw=1, ls="--", label="ES-EKF Estimate")
    axs[i].set_ylabel(vlabels[i])
axs[0].legend(loc="upper right"); axs[0].set_title("Velocity Components vs Time")
axs[2].set_xlabel("Time (s)")
savefig("04_velocity_components")

# ---------------------------------------------------------------- 5. RPY vs time
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
rlabels = ["Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
for i in range(3):
    axs[i].plot(t, np.rad2deg(eul_true[:, i]), color="black", lw=1.6, label="Ground Truth")
    axs[i].plot(t, np.rad2deg(est_eul[:, i]), color="darkorange", lw=1, ls="--", label="ES-EKF Estimate")
    axs[i].set_ylabel(rlabels[i])
axs[0].legend(loc="upper right"); axs[0].set_title("Roll / Pitch / Yaw vs Time")
axs[2].set_xlabel("Time (s)")
savefig("05_orientation_rpy")

# ---------------------------------------------------------------- 6. Position error
pos_err = est_pos - pos
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
comp_labels = ["North err (m)", "East err (m)", "Depth err (m)"]
for i in range(3):
    axs[i].plot(t, pos_err[:, i], color="crimson", lw=1)
    axs[i].axhline(0, color="gray", lw=0.7)
    sigma = np.sqrt(P_diag[:, i])
    axs[i].fill_between(t, -3*sigma, 3*sigma, color="crimson", alpha=0.15, label="$\\pm3\\sigma$ (filter)")
    axs[i].set_ylabel(comp_labels[i])
axs[0].legend(loc="upper right")
norm_err = np.linalg.norm(pos_err, axis=1)
axs[3].plot(t, norm_err, color="black", lw=1.2)
axs[3].set_ylabel("‖error‖ (m)")
axs[0].set_title("Position Estimation Error (with filter 3σ bound)")
axs[3].set_xlabel("Time (s)")
savefig("06_position_error")

# ---------------------------------------------------------------- 7. Velocity error
vel_err = est_vel - vel
fig, axs = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for i in range(3):
    axs[i].plot(t, vel_err[:, i], color="steelblue", lw=1)
    axs[i].axhline(0, color="gray", lw=0.7)
    sigma = np.sqrt(P_diag[:, 3+i])
    axs[i].fill_between(t, -3*sigma, 3*sigma, color="steelblue", alpha=0.15, label="$\\pm3\\sigma$ (filter)")
    axs[i].set_ylabel(vlabels[i].split(" ")[0] + " err (m/s)")
axs[0].legend(loc="upper right")
axs[3].plot(t, np.linalg.norm(vel_err, axis=1), color="black", lw=1.2)
axs[3].set_ylabel("‖error‖ (m/s)")
axs[0].set_title("Velocity Estimation Error (with filter 3σ bound)")
axs[3].set_xlabel("Time (s)")
savefig("07_velocity_error")

# ---------------------------------------------------------------- 8. Orientation error
eul_err_rad = np.array([np.arctan2(np.sin(est_eul[:, i]-eul_true[:, i]),
                                    np.cos(est_eul[:, i]-eul_true[:, i])) for i in range(3)]).T
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for i in range(3):
    axs[i].plot(t, np.rad2deg(eul_err_rad[:, i]), color="darkorange", lw=1)
    axs[i].axhline(0, color="gray", lw=0.7)
    sigma = np.rad2deg(np.sqrt(P_diag[:, 6+i]))
    axs[i].fill_between(t, -3*sigma, 3*sigma, color="darkorange", alpha=0.15, label="$\\pm3\\sigma$ (filter)")
    axs[i].set_ylabel(rlabels[i] + " err")
axs[0].legend(loc="upper right")
axs[0].set_title("Orientation Estimation Error (with filter 3σ bound)")
axs[2].set_xlabel("Time (s)")
savefig("08_orientation_error")

# ---------------------------------------------------------------- 9. RMSE over time (windowed)
win = 500  # samples (~5s at 100Hz)
def running_rmse(err):
    e2 = np.sum(err**2, axis=1)
    csum = np.cumsum(e2)
    rmse = np.zeros_like(csum)
    rmse[win:] = np.sqrt((csum[win:] - csum[:-win]) / win)
    rmse[:win] = np.sqrt(csum[:win] / np.arange(1, win+1))
    return rmse

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(t, running_rmse(pos_err), color="crimson")
axs[0].set_ylabel("Position RMSE (m)")
axs[0].set_title("Windowed RMSE over Time (5 s window)")
axs[1].plot(t, running_rmse(vel_err), color="steelblue")
axs[1].set_ylabel("Velocity RMSE (m/s)")
axs[1].set_xlabel("Time (s)")
savefig("09_rmse_over_time")

# ---------------------------------------------------------------- 10. Innovations
fig, axs = plt.subplots(5, 1, figsize=(10, 11), sharex=True)
names = ["depth", "dvl", "gps", "mag", "vo"]
colors = ["teal", "purple", "green", "brown", "slateblue"]
titles = ["Depth Innovation ‖y‖", "DVL Innovation ‖y‖ (m/s)", "GPS Innovation ‖y‖ (m)",
          "Magnetometer Innovation ‖y‖", "Visual Odometry Innovation ‖y‖ (m/s)"]
for i, nm in enumerate(names):
    it = extra["innovation_t"][nm]
    iv = extra["innovations"][nm]
    axs[i].scatter(it, iv, s=6, color=colors[i], alpha=0.6)
    axs[i].set_ylabel(titles[i], fontsize=8)
axs[0].set_title("Measurement Innovation Residuals (all update sources)")
axs[4].set_xlabel("Time (s)")
savefig("10_innovation_residuals")

# ---------------------------------------------------------------- 11. Covariance evolution
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
pos_sigma = np.sqrt(P_diag[:, 0:3])
vel_sigma = np.sqrt(P_diag[:, 3:6])
th_sigma = np.rad2deg(np.sqrt(P_diag[:, 6:9]))
bias_sigma_a = np.sqrt(P_diag[:, 9:12])
axs[0, 0].plot(t, pos_sigma[:, 0], label="N"); axs[0, 0].plot(t, pos_sigma[:, 1], label="E")
axs[0, 0].plot(t, pos_sigma[:, 2], label="D")
axs[0, 0].set_title("Position $\\sigma$ (m)"); axs[0, 0].legend(fontsize=8)
axs[0, 1].plot(t, vel_sigma[:, 0], label="$v_x$"); axs[0, 1].plot(t, vel_sigma[:, 1], label="$v_y$")
axs[0, 1].plot(t, vel_sigma[:, 2], label="$v_z$")
axs[0, 1].set_title("Velocity $\\sigma$ (m/s)"); axs[0, 1].legend(fontsize=8)
axs[1, 0].plot(t, th_sigma[:, 0], label="roll"); axs[1, 0].plot(t, th_sigma[:, 1], label="pitch")
axs[1, 0].plot(t, th_sigma[:, 2], label="yaw")
axs[1, 0].set_title("Attitude $\\sigma$ (deg)"); axs[1, 0].legend(fontsize=8)
axs[1, 0].set_xlabel("Time (s)")
axs[1, 1].plot(t, bias_sigma_a[:, 0], label="$b_{ax}$"); axs[1, 1].plot(t, bias_sigma_a[:, 1], label="$b_{ay}$")
axs[1, 1].plot(t, bias_sigma_a[:, 2], label="$b_{az}$")
axs[1, 1].set_title("Accel Bias $\\sigma$ (m/s²)"); axs[1, 1].legend(fontsize=8)
axs[1, 1].set_xlabel("Time (s)")
fig.suptitle("Covariance (1σ) Evolution — Uncertainty Shrinks on Sensor Updates")
savefig("11_covariance_evolution")

# ---------------------------------------------------------------- 12. Bias estimation (bonus)
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ba_true = np.array([0.08, -0.05, 0.10])
bg_true = np.array([0.004, -0.003, 0.002])
for i, lab in enumerate(["x", "y", "z"]):
    axs[0].plot(t, est_ba[:, i], label=f"$b_{{a{lab}}}$ est")
    axs[0].axhline(ba_true[i], ls=":", color=f"C{i}", alpha=0.6)
    axs[1].plot(t, est_bg[:, i], label=f"$b_{{g{lab}}}$ est")
    axs[1].axhline(bg_true[i], ls=":", color=f"C{i}", alpha=0.6)
axs[0].set_ylabel("Accel bias (m/s²)"); axs[0].legend(fontsize=8, ncol=3)
axs[1].set_ylabel("Gyro bias (rad/s)"); axs[1].legend(fontsize=8, ncol=3)
axs[1].set_xlabel("Time (s)")
axs[0].set_title("IMU Bias State Convergence (dotted = true bias)")
savefig("12_bias_convergence")

print("All plots generated in", OUT)

# ---------------------------------------------------------------- summary stats
rmse_p_total = np.sqrt(np.mean(np.sum(pos_err**2, axis=1)))
rmse_v_total = np.sqrt(np.mean(np.sum(vel_err**2, axis=1)))
rmse_yaw_deg = np.sqrt(np.mean(eul_err_rad[:, 2]**2)) * 180/np.pi
final_err = np.linalg.norm(pos_err[-1])
print(f"RMSE position: {rmse_p_total:.3f} m")
print(f"RMSE velocity: {rmse_v_total:.3f} m/s")
print(f"RMSE yaw: {rmse_yaw_deg:.3f} deg")
print(f"Final position error: {final_err:.3f} m")
