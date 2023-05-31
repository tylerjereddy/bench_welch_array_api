import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SciPy 1.10.1 wheel
numpy_wheel_timings_sec = [5.368365501053631]

# SciPy Tyler welch() branch
numpy_timings_sec = [5.911852240096778]

numpy_strict_timings_sec = [18.297413507010788]

cupy_timings_sec = [0.10812997305765748]

cupy_strict_timings_sec = [219.84708254877478]


torch_cpu_timings_sec = [0.5217398600652814]

torch_cpu_strict_timings_sec = [244.07771029276773]

torch_gpu_timings_sec = [0.08464314742013812]

torch_gpu_strict_timings_sec = [375.15701419021934]


fig, ax = plt.subplots()
x_labels = ["SciPy 1.10.1 wheel",
            "NumPy Relaxed API",
            "NumPy Strict API",
            "CuPy Relaxed API",
            "CuPy Strict API",
            "Torch CPU Relaxed API",
            "Torch CPU Strict API",
            "Torch GPU Relaxed API",
            "Torch GPU Stric API"]
for x_loc, data, color in zip(range(9),
                                      [numpy_wheel_timings_sec,
                                       numpy_timings_sec,
                                       numpy_strict_timings_sec,
                                       cupy_timings_sec,
                                       cupy_strict_timings_sec,
                                       torch_cpu_timings_sec,
                                       torch_cpu_strict_timings_sec,
                                       torch_gpu_timings_sec,
                                       torch_gpu_strict_timings_sec],
                                      ["black",
                                       "blue",
                                       "blue",
                                       "green",
                                       "green",
                                       "red",
                                       "red",
                                       "magenta",
                                       "magenta"]):
    data = np.asarray(data)
    num_trials = data.size
    if num_trials > 1:
        yerr = np.std(data)
        alpha = 1.0
    else:
        yerr = None
        alpha = 0.5
    ax.bar(x_loc,
           np.average(data),
           yerr=yerr,
           color=color,
           alpha=alpha,
           capsize=6.0)
       
ax.set_yscale("log")
ax.set_ylabel("log(Average Time from interwoven trials) (s)")
ax.set_xticks(range(9))
ax.set_xticklabels(x_labels, fontsize=8)
ax.tick_params(axis='x', rotation=90)
ax.set_title("SciPy welch() performance with array API backends")

fig.tight_layout()
fig.savefig("results.png", dpi=300)
