import pandas as pd
import matplotlib.pyplot as plt

# Path to your log file
LOG_PATH = "./submodules/Hunyuan3D_2/assets/results/Octopus/system_monitor_attempt_5.log"

# Read the log file
df = pd.read_csv(LOG_PATH, parse_dates=["Timestamp"])

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot CPU Usage %
axs[0].plot(df["Timestamp"], df["CPU_Usage_Percent"], marker='o', color='blue')
axs[0].set_title("CPU Usage (%) Over Time")
axs[0].set_ylabel("CPU Usage (%)")
axs[0].grid(True)

# Plot GPU Usage %
axs[1].plot(df["Timestamp"], df["GPU_Usage_Percent"], marker='o', color='green')
axs[1].set_title("GPU Usage (%) Over Time")
axs[1].set_ylabel("GPU Usage (%)")
axs[1].grid(True)

# Plot GPU Utilization
axs[2].plot(df["Timestamp"], df["GPU_Utilization"], marker='o', color='red')
axs[2].set_title("GPU Utilization (%) Over Time")
axs[2].set_ylabel("GPU Utilization (%)")
axs[2].set_xlabel("Timestamp")
axs[2].grid(True)

plt.tight_layout()
plt.show()
