# Reproducing the figures and animations

## Setup

Requires Python >= 3.11, < 3.13.

```bash
cd evolution-diversity
uv sync
```

## Animations

`animations/diversity_comparison.py` generates three GIFs showing evolution on the Rastrigin function under low, high, and balanced diversity:

```bash
uv run python animations/diversity_comparison.py
```

Output (saved to `figures/`):
- `diversity_low.gif` -- population collapses to a local minimum
- `diversity_high.gif` -- population wanders randomly, never converges
- `diversity_balanced.gif` -- population finds the global minimum

Pass `--show` to display interactively instead of saving.

## Static figures

`animations/figures.py` generates the remaining figures:

```bash
uv run python animations/figures.py
```

Output (saved to `figures/`):
- `phase_diagram.png` -- selection pressure vs mutation strength heatmap
- `time_series.png` -- best fitness and diversity over generations for three regimes
- `takeover_curves.png` -- takeover dynamics for different tournament sizes
- `mutation_phase_transition.png` -- optimal mutation rate curve (Witt, 2013)
- `self_adjusting_onemax.png` -- self-adjusting mutation rate on OneMax
- `markov_chain.png` -- Markov chain with and without elitism
- `crossover_cartoon.png` -- crossover with diverse vs identical parents
- `rastrigin_surface.png` -- Rastrigin function heatmap
