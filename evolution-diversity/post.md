---
layout: post
title: "Diversity Is All You Need (To Converge)"
date: 2026-04-07
author:
  name: Jack Foxabbott
  title: Founding Member of Technical Staff
  linkedin: https://www.linkedin.com/in/foxabbott/
---

*By [Jack Foxabbott](https://www.linkedin.com/in/foxabbott/), Founding Member of Technical Staff*

# Diversity Is All You Need (To Converge)

Evolutionary algorithms are simple. Maintain a population. Evaluate fitness. Keep the best. Mutate to create offspring. Repeat. Most introductions stop there.

They shouldn't. The loop isn't the hard part. **Controlling how diverse the population stays as the algorithm runs** is the hard part. Get it wrong and you either collapse to a local optimum or wander randomly forever. Get it right and you provably converge to the global optimum -- in polynomial time instead of exponential time.

We'll make that argument on a single example: an EA optimising the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function), a standard multimodal benchmark with dozens of local minima. Then we'll back it up with two theoretical results -- [Rudolph (1994)](https://doi.org/10.1109/TNNLS.1994.283510) and [Dang et al. (2016)](https://doi.org/10.1007/978-3-319-45823-6_13) -- that explain why diversity is the difference between feasible and infeasible convergence. In the [next post](../kernel-evolution/post.md), we apply these ideas to evolving GPU kernels with LLMs.

---

## The Rastrigin function

The Rastrigin function in two dimensions is `f(x, y) = 20 + (x² - 10 cos 2πx) + (y² - 10 cos 2πy)`. The global minimum is at the origin, `f(0, 0) = 0`. The landscape is covered in a regular grid of local minima separated by ridges. Any greedy, hill-climbing algorithm gets stuck. It punishes algorithms that don't explore.

```python
def rastrigin(x, y, A=10):
    return (A * 2
            + (x**2 - A * np.cos(2 * np.pi * x))
            + (y**2 - A * np.cos(2 * np.pi * y)))
```

## The EA

An evolutionary algorithm maintains a population of candidate solutions. Each generation: evaluate fitness, select the fittest as parents, mutate to create offspring. The advantage over single-point optimisation is that a population explores many regions of the search space simultaneously.

We ran the same EA on the Rastrigin function three times, varying two knobs: **selection pressure** (how aggressively we pick the best) and **mutation strength** (how far offspring can jump from their parents). Everything else was identical: 30 individuals, elitism (always keep the best), Gaussian mutations.

## Too little diversity

Tournament selection with size 10 (very aggressive -- almost always picks the single best individual). Gaussian mutations with σ = 0.15.

![Low diversity -- stuck in local optimum](figures/diversity_low.gif)

The population collapses within a few generations. Every individual clusters around the same local minimum. The mutations are too small to jump over the ridges to neighbouring basins.

## Too much diversity

Tournament selection with size 2 (almost random). Gaussian mutations with σ = 3.5.

![High diversity -- no convergence](figures/diversity_high.gif)

The opposite failure mode. The population never builds on good solutions. Each generation is essentially a fresh random sample. This is random search wearing an evolutionary costume.

## Balanced diversity

Tournament selection with size 5. Adaptive Gaussian mutations that start large and decay: `σ(t) = 1.2 * (1 - 0.8 * t/T)`.

![Balanced diversity -- finds global optimum](figures/diversity_balanced.gif)

Early on, large mutations spread the population across the landscape. As the algorithm progresses, mutations shrink, focusing the population on the best basin. The global optimum is found.

Selection *exploits*: it copies what works. Mutation *explores*: it tries something new. The balanced case transitions from broad exploration to focused exploitation. The other two cases get stuck at one extreme.

## Why this works: two theorems

### Rudolph (1994): convergence with probability 1

[Rudolph (1994)](https://doi.org/10.1109/TNNLS.1994.283510) proved that an EA converges to the global optimum with probability 1 if three conditions hold:

1. **Elitism.** Never discard the best solution.
2. **Ergodicity.** The mutation operator can eventually reach any valid solution from any starting point.
3. **Selection pressure.** Better solutions are more likely to survive.

These conditions are mild. Elitism is a one-line code change. Gaussian mutation is ergodic by construction. Tournament selection satisfies the third.

The catch: the theorem says nothing about how long convergence takes. It could be exponential.

### Dang et al. (2016): diversity makes it polynomial

[Dang, Jansen, Lehre, Oliveto, Sudholt et al. (2016)](https://doi.org/10.1007/978-3-319-45823-6_13) studied how long it takes an EA to escape a local optimum. Their result: **a single candidate takes exponential time. A population that maximises spread between candidates does it in polynomial time.**

This isn't a constant-factor speedup. It's the difference between an algorithm that terminates and one that doesn't, on any practical time horizon.

The intuition: a single candidate in a local optimum needs a mutation that jumps it all the way to a better basin. The probability is exponentially small in the distance. A diverse population has individuals scattered near multiple basin boundaries. At least one is likely close enough that a modest mutation tips it over the ridge. The population parallelises the escape.

## Mapping the example to the theory

| Scenario | Elitism | Ergodicity | Selection | Diversity | Outcome |
|----------|---------|------------|-----------|-----------|---------|
| Too little | Yes | Technically, but σ too small | Very strong | Collapsed | Local optimum |
| Too much | Yes | Yes | Too weak | Maximal, unstructured | Random walk |
| Balanced | Yes | Yes | Moderate | Controlled, decaying | Global optimum |

In the first case, Rudolph's conditions are technically met, but Dang et al. tells us why it fails in practice: the population has collapsed, so escaping requires an exponentially unlikely mutation.

In the second case, diversity is maximal but selection is too weak to exploit discoveries. Rudolph's conditions hold but the algorithm never builds on what it finds.

In the third case, the population starts diverse (polynomial escape time per Dang et al.) and gradually focuses. All three conditions are satisfied *and* diversity is managed so convergence happens in reasonable time.

## The point

Evolutionary algorithms are not random search. They have provable convergence guarantees. But those guarantees are vacuous without diversity management.

Too little diversity: stuck. Too much: lost. The science is in controlling the transition from exploration to exploitation as the algorithm runs. This has consequences beyond toy functions. In the [next post](../kernel-evolution/post.md), we apply these principles to evolving GPU kernels, where the mutation operator is an LLM and the search space is code.

---

*All code to generate the animations in this post is in [`animations/diversity_comparison.py`](animations/diversity_comparison.py). The benchmark data are available at [github.com/GeometricAGI/blog](https://github.com/GeometricAGI/blog).*

## References

1. G. Rudolph. *Convergence Analysis of Canonical Genetic Algorithms.* IEEE Transactions on Neural Networks, 5(1):96--101, 1994. [doi:10.1109/TNNLS.1994.283510](https://doi.org/10.1109/TNNLS.1994.283510)

2. D.-C. Dang, T. Jansen, P.K. Lehre, P.S. Oliveto, D. Sudholt. *Escaping Local Optima using Crossover with Mutation.* Parallel Problem Solving from Nature (PPSN XIV), pp. 160--170, 2016. [doi:10.1007/978-3-319-45823-6_13](https://doi.org/10.1007/978-3-319-45823-6_13)
