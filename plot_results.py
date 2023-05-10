import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

numpy_timings_sec = [4.919703705934808,
        5.045966051053256,
        4.901492156088352,
        4.895045991055667,
        4.9723517938982695,
        ]

numpy_strict_timings_sec = [15.710065103136003]

cupy_timings_sec = [0.23821475682780147,
        0.24966995394788682,
        0.24989571701735258,
        0.25607424485497177,
        0.24746034992858768,
        ]

cupy_strict_timings_sec = [155.41026511299424]


torch_cpu_timings_sec = [
        1.1638194161932915,
        1.120931203942746,
        1.0023795650340617,
        1.1782302970532328,
        1.1800141278654337,
        ]

torch_cpu_strict_timings_sec = [
        180.73112939088605]

torch_gpu_timings_sec = [
        0.13685095217078924,
        0.1785686588846147,
        0.17490072012878954,
        0.13612257200293243,
        0.1353681399486959,
        ]

torch_gpu_strict_timings_sec = [
        281.28290371201,
    ]


fig, ax = plt.subplots()
x_labels = ["NumPy Relaxed API",
            "NumPy Strict API",
            "CuPy Relaxed API",
            "CuPy Strict API",
            "Torch CPU Relaxed API",
            "Torch CPU Strict API",
            "Torch GPU Relaxed API",
            "Torch GPU Stric API"]
for x_loc, data, color in zip(range(8),
                                      [numpy_timings_sec,
                                       numpy_strict_timings_sec,
                                       cupy_timings_sec,
                                       cupy_strict_timings_sec,
                                       torch_cpu_timings_sec,
                                       torch_cpu_strict_timings_sec,
                                       torch_gpu_timings_sec,
                                       torch_gpu_strict_timings_sec],
                                      ["blue",
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
ax.set_xticks(range(8))
ax.set_xticklabels(x_labels, fontsize=8)
ax.tick_params(axis='x', rotation=90)
ax.set_title("SciPy welch() performance with array API backends")

fig.tight_layout()
fig.savefig("results.png", dpi=300)
