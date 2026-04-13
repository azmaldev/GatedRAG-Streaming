# GatedRAG-Streaming: Parallel Retrieval During Token Generation

## Overview

GatedRAG-Streaming is an evolution of GatedRAG that performs retrieval in parallel with token generation, achieving significantly lower latency through async/background retrieval.

## Core Innovation

**Previous (Sequential):**
```
Generate → Post-process → Retrieve → Inject → Done
Total: 7.5s
```

**New (Parallel):**
```
Generate token 1 → (async) Retrieve token 1 (background) → Generate token 2 → (async) Retrieve token 2 → Inject when ready → Stream output
Total: 1.5-1.8s
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              GatedRAG-Streaming                           │
├─────────────────────────────────────────────────────────────┤
│  Prompt → ┌──────────────┐                             │
│           │ Gate Network │ → Confidence Score              │
│           └──────────────┘                             │
│                ↓                                        │
│    ┌─────────────┴─────────────┐                       │
│    ↓                           ↓                       │
│ Generation Thread        Retrieval Thread                │
│ (Main - Streaming)       (Background - daemon)            │
│    ↓                           ↓                       │
│    └─────────────┬─────────────┘                       │
│                  ↓                                     │
│           Queue (non-blocking)                         │
│                  ↓                                     │
│           Cache (thread-safe)                          │
│                  ↓                                     │
│           Stream Output                               │
└─────────────────────────────────────────────────────────────┘
```

## Performance

| System | Latency | Notes |
|--------|---------|-------|
| RAG Baseline | 1.36s | Sequential retrieval |
| GatedRAG | 7.5s | Post-processing |
| GatedRAG-Streaming | 1.5-1.8s | Parallel retrieval |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from gated_rag_streaming import GatedRAGStreaming

gated_rag = GatedRAGStreaming()

prompt = "The capital of France is Paris. It is known for"
output, metrics = gated_rag.generate(prompt, use_gating=True)

print(f"Output: {output}")
print(f"Total time: {metrics['total_time']:.3f}s")
```

## Quick Start

```bash
python gated_rag_streaming.py
python benchmarks/benchmark_streaming.py
```

## Based On

- GatedRAG (https://github.com/azmaldev/GatedRAG)
- 2025 AI breakthroughs: Tongyi DeepResearch, Tok-RAG, DioR

## License

MIT