from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Radice del progetto = cartella sopra implementation/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data" / "raw"

classes = ["glioma", "meningioma", "notumor", "pituitary"]
labels  = ["Glioma", "Meningioma", "No tumor", "Pituitary"]
exts = {".jpg", ".jpeg", ".png"}

for split in ("Training", "Testing"):
    for c in classes:
        p = ROOT / split / c
        if not p.is_dir():
            raise FileNotFoundError(f"Cartella non trovata: {p}")

def count(split):
    return [
        sum(1 for f in (ROOT / split / c).iterdir() if f.suffix.lower() in exts)
        for c in classes
    ]

train = count("Training")
test  = count("Testing")

print(f"{'Class':12s} {'Train':>6s} {'Test':>6s}")
for l, a, b in zip(labels, train, test):
    print(f"{l:12s} {a:6d} {b:6d}")
print(f"{'Total':12s} {sum(train):6d} {sum(test):6d}")

x = np.arange(len(classes))
w = 0.38
fig, ax = plt.subplots(figsize=(6, 3.5))
b1 = ax.bar(x - w/2, train, w, label="Training")
b2 = ax.bar(x + w/2, test,  w, label="Testing")
ax.bar_label(b1, padding=2, fontsize=8)
ax.bar_label(b2, padding=2, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Number of images")
ax.set_xlabel("Class")
ax.legend()
plt.tight_layout()

out = PROJECT_ROOT / "fig4_class_distribution.png"
plt.savefig(out, dpi=300)
print("Figura salvata in:", out)
plt.show()