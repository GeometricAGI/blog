---
layout: post
title: "Evolving GPU Kernels with LLMs"
date: 2026-04-07
author:
  name: Jack Foxabbott
  title: Founding Member of Technical Staff
  linkedin: https://www.linkedin.com/in/foxabbott/
---

*By [Jack Foxabbott](https://www.linkedin.com/in/foxabbott/), Founding Member of Technical Staff*

# Evolving GPU Kernels with LLMs

*How we use large language models as mutation operators in an evolutionary algorithm to discover optimised GPU kernels -- and why controlling diversity is still the key.*

---

In the [previous post](../evolution-diversity/post.md), we showed that diversity is the secret ingredient in evolutionary algorithms. Too little and the population collapses to a local optimum. Too much and it wanders randomly. Get the balance right and you provably converge to the global optimum in polynomial time.

That was on a continuous function in $\mathbb{R}^2$. Now the individuals are not points on a heatmap but *programs* -- specifically, GPU kernels -- and the search space is the space of all valid code.

The diversity story carries over, and it matters more here: the search space is harder, evaluation is expensive, and most random changes to code are destructive.

## The Problem: GPU Kernels Are Hard

Writing fast GPU kernels is one of the most demanding tasks in software engineering. Everything is coupled: tiling strategy, occupancy, register pressure, memory access patterns. A small change to one dimension can cause a performance cliff in another. And what's optimal on an H200 may be suboptimal on a B200.

The search space is combinatorial. For any given operation, you're choosing among data types, input shapes, target devices, and an infinite space of algorithmic rewrites. Layer on multiple objectives -- latency, memory usage, numerical accuracy -- and you have a Pareto frontier, not a single answer.

This is exactly the kind of problem evolutionary algorithms are built for: huge, multi-modal search spaces where gradient information is unavailable. But there's a catch.

## Classical Genetic Programming Breaks Code

The traditional approach to evolving programs is [genetic programming](https://en.wikipedia.org/wiki/Genetic_programming) (GP): represent programs as syntax trees and apply random structural mutations -- swap a subtree, change an operator, delete a node.

The problem is that random structural mutations almost always produce broken code. Here's a simple example: take a function that computes Euclidean distance and apply five classical GP mutations to it.

```python
def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return sqrt(dx*dx + dy*dy)
```

| Mutation | What It Does | Result |
|----------|-------------|--------|
| Point mutation | Replace `sqrt` with `[qrt` | `SyntaxError: invalid syntax` |
| Subtree crossover | Swap line order | `IndentationError: unexpected indent` |
| Subtree mutation | Replace `dy*dy` with `len([y2])` | Wrong result: 3.16 (expected 5.0) |
| Operator mutation | Replace `-` with `>>` | Wrong result: 4.0 (expected 5.0) |
| Node deletion | Remove `dy = y2 - y1` | `NameError: name 'dy' is not defined` |

**5 out of 5 mutations break the code.** This isn't bad luck -- it's the norm. Most of the search budget in classical GP is wasted on programs that don't even parse, let alone produce correct results.

![Classical GP mutations breaking code](figures/classical_gp.gif)

In the language of the previous post: classical GP mutations destroy the *ergodicity* condition in practice. Technically, random mutations *can* reach any program given infinite time. But the probability of a useful mutation is so vanishingly small that you'd need exponential time to find one -- exactly the regime [Dang et al. (2016)](https://doi.org/10.1145/2908812.2908956) showed leads to exponential escape times.

## LLMs: The Ideal Mutation Operator

Large language models fix this. Instead of random structural perturbations, an LLM can:

- **Understand code semantics**, so mutations are *meaningful*, not random.
- **Read documentation and hardware specs** to guide changes toward known optimisation strategies.
- **Produce code that compiles and runs.** Most LLM-generated mutations are syntactically valid.
- **Control diversity directly via temperature.** Low temperature produces conservative edits; high temperature produces radical rewrites.

This idea -- using LLMs as variation operators in an evolutionary loop -- has strong recent precedent. [Lehman et al. (2022)](https://arxiv.org/abs/2206.08896) introduced Evolution through Large Models (ELM), showing that LLMs trained on code can serve as mutation operators for genetic programming, generating hundreds of thousands of functional programs in a domain absent from training data. [Meyerson et al. (2023)](https://arxiv.org/abs/2302.12170) formalised why this works: LLM-based crossover implicitly builds a probabilistic model of parent genotypes and samples offspring, connecting it to Estimation of Distribution Algorithms. The pre-trained distribution concentrates mass on syntactically valid, semantically coherent programs, so most mutations stay in a productive neighbourhood.

Google DeepMind's [FunSearch (Romera-Paredes et al., 2024)](https://www.nature.com/articles/s41586-023-06924-6) applied this to mathematical discovery, pairing an LLM with an evaluator in an island-based evolutionary loop to find new constructions for the cap set problem. Its successor [AlphaEvolve (Novikov et al., 2025)](https://arxiv.org/abs/2506.13131) operates directly on code diffs and achieved a 23% speedup on a critical Gemini training kernel -- the closest published precedent to what we do.

In evolutionary terms, LLMs give you **structured diversity**: meaningful variation in algorithmic choices while maintaining syntactic stability. They satisfy the ergodicity condition *practically*, not just theoretically. An LLM can generate any valid kernel, and it does so with high enough probability that escape from local optima happens in reasonable time.

## Why Kernels Are Especially Hard

Even with LLMs as the mutation operator, kernel optimisation presents unique challenges:

- **Compilation cost.** Each candidate must compile, often with expensive autotuning. You can't evaluate thousands of candidates cheaply.
- **Hardware specificity.** The optimal kernel for one GPU architecture may be suboptimal for another.
- **Correctness validation.** Fast but wrong is worthless. Every variant needs numerical verification against a reference implementation.
- **Multiple objectives.** Speed, memory, and numerical precision are all in tension.
- **DSL complexity.** GPU domain-specific languages (Triton, CuTe/CUTLASS, Helion) are niche and under-represented in LLM training data. Mutations need documentation-grounded guidance, not just pattern matching.

Every evaluation is expensive, which means every mutation must count. This makes diversity management even more critical than in the Rastrigin example: you can't afford to waste your evaluation budget on redundant or random candidates.

## Our System: Neural Kernel Search

The system follows a standard evolutionary loop, with LLM-powered mutation:

```
Population --> LLM Mutation --> Validation --> Profiling --> Selection
    ^                                                          |
    +----------------------------------------------------------+
```

### The LLM Mutation Pipeline

Each mutation attempt passes through a multi-stage pipeline:

```
Plan --> Doc Read --> Coding --> Compile --> Correctness --> Profile
                                   |             |
                                   v             v
                                 [fail]        [fail]
                                   \            /
                                    --> Debug ---> (retry Coding)
```

1. **Plan.** The LLM generates a high-level optimisation strategy for this mutation (e.g., "try TMA async loads to overlap memory transfers with computation").
2. **Doc Read.** The LLM reads relevant documentation for the target DSL and hardware.
3. **Coding.** The LLM writes the mutated kernel.
4. **Compile.** The kernel is compiled. If compilation fails, a debug agent diagnoses the error and retries from the coding stage.
5. **Correctness.** The kernel is tested against a reference implementation. If it produces wrong results, the debug agent retries.
6. **Profile.** The kernel is benchmarked. Fitness = geometric mean speedup vs `torch.compile` across a range of input shapes.

### Controlling Diversity: Five Knobs

Just as in the Rastrigin example, diversity is controlled by tuning multiple knobs simultaneously. Here are the five we use:

| Knob | Low Diversity | High Diversity |
|------|--------------|----------------|
| **Selection pressure** | Only top-1 parent | Uniform random parent |
| **Population size** | Few candidates, many generations | Many candidates, fewer generations |
| **LLM temperature** | Low (conservative edits) | High (radical rewrites) |
| **Planning prompts** | Single plan shared across the generation | Unique plan per attempt |
| **Insight sharing** | Full history shared with all agents | No sharing between agents |

The strategy: **explore broadly early, exploit the best later.** In early generations, we use high temperature, diverse planning prompts, and weak selection to maximise the variety of kernel strategies explored. In later generations, we tighten selection, lower temperature, and share insights, focusing the population on refining the most promising approaches.

One open problem: **measuring diversity of kernels is not well-defined.** For points in $\mathbb{R}^2$, we can compute pairwise distances. For kernel code, it's much less clear. AST distance? Algorithmic similarity? Performance profile distance? There's no consensus, and this remains an active research question.

## Results: 1.6--1.8x Faster Than torch.compile

We used this system to evolve a kernel for a custom neural architecture on NVIDIA H100 GPUs. The target operation was a fused LoRA + RoPE kernel -- something no standard library provides and `torch.compile` can't specialise for.

![Speedup chart: custom kernel vs torch.compile on H100](figures/speedup_chart.png)

Across five input shapes, the evolved kernel consistently achieves **1.6--1.8x speedup** over `torch.compile`:

| Input Shape | torch.compile (ms) | Evolved Kernel (ms) | Speedup |
|-------------|--------------------|--------------------|---------|
| 8 x 512     | 0.087              | 0.054              | 1.62x   |
| 16 x 1K     | 0.319              | 0.177              | 1.80x   |
| 32 x 2K     | 1.165              | 0.686              | 1.70x   |
| 64 x 4K     | 4.551              | 2.680              | 1.70x   |
| 128 x 8K    | 18.128             | 10.653             | 1.70x   |

### What the System Discovered

Starting from a naive online softmax kernel with direct global memory loads and scalar accumulation, the system discovered Hopper-specific optimisations including:

- **TMA async bulk loads** -- overlapping global-to-shared memory transfers with computation.
- **Swizzled shared memory layouts** compatible with WGMMA (Warp Group Matrix Multiply-Accumulate).
- **Barrier-based synchronisation** using `mbarrier` instead of `__syncthreads`.

![Code diff: before and after evolution](figures/code_diff.png)

The core math is identical. The data movement is completely different. This required deep knowledge of Hopper's memory hierarchy that the LLM discovered through iterative mutation and selection -- not through a single-shot prompt.

## Connecting Back to Theory

Recall [Rudolph's (1994)](https://doi.org/10.1109/72.265964) convergence result: an EA with elitism and an ergodic mutation operator converges to the global optimum almost surely. Our system satisfies both conditions:

| Condition | How our system satisfies it |
|-----------|-----------------------------|
| **Elitism** | We always keep the best kernels across generations. |
| **Ergodicity** | LLMs can generate any valid kernel -- and unlike random mutation, they do so with practical probability. |

And recall [Dang et al.'s (2016, 2018)](https://doi.org/10.1109/TEVC.2017.2724201) result: on problems with deceptive local optima, a mutation-only EA needs exponential time to escape, while a diverse population with crossover does it in polynomial time.

The LLM makes ergodicity *practical* -- it concentrates mutations on promising directions while maintaining the theoretical ability to reach any valid kernel. And the diversity knobs (temperature, planning prompts, insight sharing) give us direct control over the exploration--exploitation tradeoff that [Dang et al.](https://doi.org/10.1145/2908812.2908956) showed determines whether convergence is polynomial or exponential.

How we manage diversity determines how fast we converge. The theoretical guarantee of eventual convergence (Rudolph) is only useful if convergence time is reasonable (Dang et al.). And convergence time is only reasonable if the population maintains the right level of diversity at the right time.

## What's Next

Kernel evolution is still early. Some open directions:

- **Better diversity metrics for code.** How do you measure the "spread" of a population of kernel implementations? Solving this would let us adaptively control diversity the way we adaptively control mutation strength in the Rastrigin example.
- **Multi-objective Pareto evolution.** Currently we optimise primarily for latency. Extending to explicit Pareto frontiers over latency, memory, and numerical precision would better match real deployment constraints.
- **Transfer across architectures.** Can a population evolved on H100 seed the search on B200, bootstrapping the process instead of starting from scratch?

Whether the individuals are points in $\mathbb{R}^2$ or GPU kernels in CuTe DSL, the same principle applies: **controlling diversity is the key to efficient evolutionary search.** LLMs make it practical.

---

*All code to generate the figures in this post is available at [github.com/GeometricAGI/blog](https://github.com/GeometricAGI/blog).*

## References

1. G. Rudolph. *Convergence Analysis of Canonical Genetic Algorithms.* IEEE Transactions on Neural Networks, 5(1):96--101, 1994. [doi:10.1109/72.265964](https://doi.org/10.1109/72.265964)

2. D.-C. Dang, T. Friedrich, M. Kötzing, M.S. Krejca, P.K. Lehre, P.S. Oliveto, D. Sudholt, A.M. Sutton. *Escaping Local Optima with Diversity Mechanisms and Crossover.* GECCO 2016, pp. 645--652. [doi:10.1145/2908812.2908956](https://doi.org/10.1145/2908812.2908956)

3. D.-C. Dang et al. *Escaping Local Optima using Crossover with Emergent Diversity.* IEEE Transactions on Evolutionary Computation, 22(3):484--497, 2018. [doi:10.1109/TEVC.2017.2724201](https://doi.org/10.1109/TEVC.2017.2724201)

4. J. Lehman, J. Gordon, S. Jain, K. Ndousse, C. Yeh, K.O. Stanley. *Evolution through Large Models.* arXiv:2206.08896, 2022.

5. E. Meyerson, M.J. Nelson, H. Bradley, A. Gaier, A. Moradi, A.K. Hoover, J. Lehman. *Language Model Crossover: Variation through Few-Shot Prompting.* ACM Transactions on Evolutionary Learning and Optimization, 2024. [arXiv:2302.12170](https://arxiv.org/abs/2302.12170)

6. B. Romera-Paredes et al. *Mathematical Discoveries from Program Search with Large Language Models.* Nature, 625:468--475, 2024.

7. A. Novikov et al. *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery.* arXiv:2506.13131, 2025.

8. J.R. Koza. *Genetic Programming: On the Programming of Computers by Means of Natural Selection.* MIT Press, 1992.

9. A. Ouyang et al. *KernelBench: Can LLMs Write Efficient GPU Kernels?* ICML 2025. [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)
