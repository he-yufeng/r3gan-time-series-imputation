"""Generate Figure 1: Refinement Effectiveness vs Coarse Quality."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
data = [
    # (dataset, coarse_method, before_MAE, after_MAE)
    ('Weather', 'Zero fill', 0.728, 0.228),
    ('Weather', 'Mean fill', 0.728, 0.223),
    ('Weather', 'Linear interp', 0.067, 0.067),
    ('Electricity', 'Zero fill', 0.832, 0.427),
    ('Electricity', 'Mean fill', 0.831, 0.429),
    ('Electricity', 'Linear interp', 0.164, 0.164),
    ('AirQuality', 'Zero fill', 0.765, 0.228),
    ('AirQuality', 'Linear interp', 0.151, 0.152),
]

# Also add SOTA baselines for context (Weather only)
baselines = [
    ('BRITS', 0.039),
    ('SAITS', 0.062),
]

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# Color and marker by dataset
ds_style = {
    'Weather': ('tab:blue', 'o'),
    'Electricity': ('tab:orange', 's'),
    'AirQuality': ('tab:green', '^'),
}

# Plot diagonal (no improvement line)
lims = [0, 0.9]
ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='No improvement')

# Plot each point
for ds, coarse, before, after in data:
    color, marker = ds_style[ds]
    label_text = None
    # Only add dataset label once
    if coarse == 'Zero fill':
        label_text = ds
    ax.scatter(before, after, c=color, marker=marker, s=80, zorder=5,
               label=label_text, edgecolors='black', linewidths=0.5)
    # Annotate coarse method
    offset = (5, 5) if before > 0.5 else (5, -12)
    fontsize = 7
    ax.annotate(coarse.split()[0], (before, after), textcoords="offset points",
                xytext=offset, fontsize=fontsize, color=color, alpha=0.8)

# Shade the "refinement effective" region (below diagonal)
ax.fill_between(lims, [0, 0], lims, alpha=0.05, color='green')
ax.fill_between(lims, lims, [1, 1], alpha=0.05, color='red')

# Add text annotations for regions
ax.text(0.55, 0.15, 'GAN helps\n(48-70% improvement)', fontsize=8,
        color='green', alpha=0.7, ha='center', style='italic')
ax.text(0.15, 0.2, 'GAN hurts', fontsize=8,
        color='red', alpha=0.5, ha='center', style='italic')

# Add BRITS/SAITS reference lines
for name, mae in baselines:
    ax.axhline(y=mae, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    ax.text(0.82, mae + 0.008, name, fontsize=7, color='gray', alpha=0.7)

ax.set_xlabel('Coarse Imputation MAE (before refinement)', fontsize=10)
ax.set_ylabel('R3GAN-1D Refined MAE (after)', fontsize=10)
ax.set_xlim(0, 0.9)
ax.set_ylim(0, 0.5)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Adversarial Refinement Effectiveness', fontsize=11)

plt.tight_layout()
plt.savefig('refinement_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('refinement_plot.png', dpi=150, bbox_inches='tight')
print('Figure saved: refinement_plot.pdf / .png')
