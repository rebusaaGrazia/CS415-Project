import matplotlib.pyplot as plt
import numpy as np

placements = ["Early", "Middle", "Late"]
se   = [580, 2184, 33312]
cbam = [610, 2146, 32866]
x = np.arange(3); w = 0.35

plt.bar(x - w/2, se, w, label="SE")
plt.bar(x + w/2, cbam, w, label="CBAM")
plt.yscale("log")
plt.xticks(x, placements)
plt.ylabel("Added parameters (log scale)")
plt.xlabel("Placement")
plt.legend()
plt.tight_layout()
plt.savefig("fig3_param_overhead.png", dpi=300)