# TASC

Modular language model: one frozen base model plus a library of small skill adapters. For each request the system decides up front which adapter it needs, or whether the base model can answer on its own.

## Why

A single model that is good at everything is expensive to train and to serve. Adapters are cheap, but a system with many of them has a practical problem: you do not know in advance which ones a request will need, so you either load all of them or guess and lose quality.

TASC trains the adapters and the router together so that the right adapter can be picked from the start of the request, before generation, with little loss compared to running with all adapters. Simple requests (greetings, general questions) go to the base model with no adapter at all.

What this gives in practice:

- one base model serves many skills;
- per request you pay for the base model plus at most one small adapter;
- new skills are added as new adapters, the base model is never retrained.

## How it works (high level)

1. The base model reads the request. Its weights are frozen and are verified unchanged by hash after every run.
2. A lightweight router looks at the start of the request and picks one adapter or none.
3. The base model generates the answer with the chosen adapter active.

The training method that makes the choice in step 2 predictable is our own work. It is not described or included in this repository.

## Current setup

| | |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct, frozen (1.5B planned) |
| Adapters | LoRA rank 16 on all attention and MLP projections, about 8.8M parameters each (about 1.8% of the base) |
| Skills | code, math, writing, plus a "general" group where the correct choice is no adapter |
| Evaluation | GSM8K, MBPP, HumanEval (generation, answers checked by running tests or matching the final number), WikiText-103 and Dolly (held-out loss) |
| Baselines | standard multi-adapter training with a router, per-domain adapters with a domain classifier, entropy gating, best single adapter, random choice, hindsight oracle |
| Statistics | paired comparison on the same test examples, several seeds, bootstrap intervals over seeds and examples |

## Status

- Training and evaluation pipeline (PyTorch, Hugging Face) is built and covered by tests.
- A short end-to-end run of all training variants was completed on an NVIDIA L4: base model unchanged, adapters reduce loss on their domains.
- Small-scale controlled experiments were encouraging. The full comparison on the real model has not been run yet.

Measured on one NVIDIA L4 (bf16):

| Workload | Result |
|---|---|
| Adapter training step, batch 8, 384 tokens | 0.36 s |
| Greedy generation, batch 256, up to 320 new tokens | 6.1 prompts/s |

## Roadmap

1. Port the pipeline to ROCm and validate it on AMD Instinct MI300X.
2. Run the full comparison against all baselines over 3 to 5 seeds.
3. Scale the base model to 1.5B.
4. Publish ROCm porting notes and MI300X vs L4 throughput numbers for multi-adapter LoRA training and generation.
5. Adapter sizing: how large an adapter each skill actually needs.

## What will be published here

Benchmarks, porting notes and aggregate results. The training method and its source code stay private.

## License

All rights reserved unless a file states otherwise.
