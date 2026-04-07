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

We'll make that argument on a single example: an EA optimising the [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function), a standard multimodal benchmark with dozens of local minima. Then we'll back it up with two theoretical results -- [Rudolph (1994)](https://doi.org/10.1109/72.265964) and [Dang et al. (2016)](https://doi.org/10.1109/TEVC.2017.2724201) -- that explain why diversity is the difference between feasible and infeasible convergence. In the [next post](../kernel-evolution/post.md), we apply these ideas to evolving GPU kernels with LLMs.

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

The decaying mutation schedule has theoretical support: [Doerr, Giessen, and Witt (2019)](https://doi.org/10.1007/s00453-018-0502-x) proved that self-adjusting mutation rates -- creating offspring at both higher and lower rates, then adapting toward whichever works better -- can achieve asymptotically optimal runtime on standard benchmarks. Our simple decay schedule approximates this: large steps when the population needs to spread, small steps when it needs to refine.

## Why this works: two theorems

### Rudolph (1994): convergence with probability 1

[Rudolph (1994)](https://doi.org/10.1109/72.265964) used Markov chain analysis to prove two things about canonical genetic algorithms. First, a negative result: without elitism, the standard GA **never** converges to the global optimum, regardless of initialisation, crossover operator, or objective function. The population's state space is ergodic -- it visits optimal states infinitely often but leaves them infinitely often. There are no absorbing states.

Second, a positive result: adding **elitism** (always preserving the best solution found so far) makes the set of populations containing a global optimum into an absorbing set. Combined with **ergodicity** of the mutation operator (any solution is reachable from any starting point with nonzero probability), the algorithm converges to the global optimum almost surely. He later extended this from finite search spaces to arbitrary metric spaces in [Rudolph (1996)](https://ieeexplore.ieee.org/document/542332/), which won the best paper award at IEEE ICEC.

The conditions are mild. Elitism is a one-line code change. Gaussian mutation is ergodic by construction. But the theorem says nothing about how long convergence takes -- it could be exponential. Selection pressure affects *speed*, not the convergence guarantee itself.

### Dang et al. (2016, 2018): diversity makes it polynomial

[Dang, Friedrich, Kötzing, Krejca, Lehre, Oliveto, Sudholt, and Sutton (2016)](https://doi.org/10.1145/2908812.2908956) studied how long it takes an EA to escape a local optimum, using the $\text{Jump}_k$ family of fitness functions -- a standard theoretical benchmark where a gap of width $k$ surrounds the global optimum, creating a deceptive local attractor. They proved that a mutation-only (1+1) EA needs $\Theta(n^k)$ evaluations to cross the gap. When $k$ grows with problem size $n$, this is exponential.

But a population-based GA with crossover and diversity mechanisms -- such as deterministic crowding, fitness sharing, or island models -- solves the same problem in $O(n \log n)$. The improvement comes from crossover recombining genetically diverse parents: one parent might have the right bits on the left, the other on the right, and crossover assembles the solution that neither parent could reach alone.

The catch is that crossover only helps if the population is diverse. Without diversity mechanisms, the population collapses and crossover recombines identical parents, doing nothing. The [journal version (Dang et al., 2018)](https://doi.org/10.1109/TEVC.2017.2724201) studies seven diversity mechanisms and shows that all of them enable the exponential-to-polynomial speedup, though through different mechanisms: some maintain diversity explicitly (fitness sharing, island models), while others allow diversity to "emerge" from the interaction of crossover and selection.

For a comprehensive survey of these results, see [Sudholt (2018)](https://arxiv.org/abs/1801.10087).

The intuition carries beyond $\text{Jump}_k$. A single candidate in a local optimum needs a mutation that jumps it all the way to a better basin. The probability is exponentially small in the distance. A diverse population has individuals scattered near multiple basin boundaries. At least one is likely close enough that a modest mutation -- or a crossover with a distant partner -- tips it over the ridge. The population parallelises the escape.

## Mapping the example to the theory

| Scenario | Elitism | Ergodicity | Selection | Diversity | Outcome |
|----------|---------|------------|-----------|-----------|---------|
| Too little | Yes | Technically, but σ too small | Very strong | Collapsed | Local optimum |
| Too much | Yes | Yes | Too weak | Maximal, unstructured | Random walk |
| Balanced | Yes | Yes | Moderate | Controlled, decaying | Global optimum |

In the first case, Rudolph's conditions are technically met (Gaussian noise is ergodic, elitism is on), so convergence is guaranteed -- eventually. But Dang et al. tells us why "eventually" is useless: the population has collapsed to a single basin, so escaping the local optimum requires an exponentially unlikely mutation. The convergence guarantee holds in theory but not on any finite time budget.

In the second case, diversity is maximal but selection is too weak to exploit discoveries. Rudolph's conditions hold, but the algorithm never builds on what it finds. It converges in the limit but makes no practical progress.

In the third case, the population starts diverse (enabling the polynomial escape time that Dang et al. proved) and gradually focuses. Both Rudolph's convergence conditions and Dang et al.'s diversity conditions are satisfied, so convergence happens in reasonable time.

## The point

Evolutionary algorithms are not random search. They have provable convergence guarantees. But those guarantees are vacuous without diversity management.

Too little diversity: stuck. Too much: lost. The science is in controlling the transition from exploration to exploitation as the algorithm runs. This has consequences beyond toy functions. In the [next post](../kernel-evolution/post.md), we apply these principles to evolving GPU kernels, where the mutation operator is an LLM and the search space is code.

---

*All code to generate the animations in this post is in [`animations/diversity_comparison.py`](animations/diversity_comparison.py). The benchmark data are available at [github.com/GeometricAGI/blog](https://github.com/GeometricAGI/blog).*

## References

1. G. Rudolph. *Convergence Analysis of Canonical Genetic Algorithms.* IEEE Transactions on Neural Networks, 5(1):96--101, 1994. [doi:10.1109/72.265964](https://doi.org/10.1109/72.265964)

2. G. Rudolph. *Convergence of Evolutionary Algorithms in General Search Spaces.* Proceedings of the 3rd IEEE Conference on Evolutionary Computation (ICEC), pp. 50--54, 1996. Best paper award.

3. D.-C. Dang, T. Friedrich, M. Kötzing, M.S. Krejca, P.K. Lehre, P.S. Oliveto, D. Sudholt, A.M. Sutton. *Escaping Local Optima with Diversity Mechanisms and Crossover.* GECCO 2016, pp. 645--652. [doi:10.1145/2908812.2908956](https://doi.org/10.1145/2908812.2908956)

4. D.-C. Dang, T. Friedrich, M. Kötzing, M.S. Krejca, P.K. Lehre, P.S. Oliveto, D. Sudholt, A.M. Sutton. *Escaping Local Optima using Crossover with Emergent Diversity.* IEEE Transactions on Evolutionary Computation, 22(3):484--497, 2018. [doi:10.1109/TEVC.2017.2724201](https://doi.org/10.1109/TEVC.2017.2724201)

5. T. Friedrich, P.S. Oliveto, D. Sudholt, C. Witt. *Analysis of Diversity-Preserving Mechanisms for Global Exploration.* Evolutionary Computation, 17(4):455--476, 2009. [doi:10.1162/evco.2009.17.4.17401](https://doi.org/10.1162/evco.2009.17.4.17401)

6. B. Doerr, C. Giessen, C. Witt. *The (1+λ) Evolutionary Algorithm with Self-Adjusting Mutation Rate.* Algorithmica, 81:593--631, 2019. [doi:10.1007/s00453-018-0502-x](https://doi.org/10.1007/s00453-018-0502-x)

7. D. Sudholt. *The Benefits of Population Diversity in Evolutionary Algorithms: A Survey of Rigorous Runtime Analyses.* arXiv:1801.10087, 2018.
