# GatedRAG-Streaming Architecture

## Why Streaming + Parallel is Better

Traditional RAG (sequential):
```
1. Generate all tokens (slow)
2. Post-process (find [RETRIEVE: queries])
3. Sequential retrieval (one at a time)
4. Regenerate with context
Total: ~7.5s (or more)
```

Streaming + Parallel:
```
1. Generate tokens one-by-one (streaming)
2. For each token:
   - Check gate score
   - If gate_score < 0.6: queue retrieval (non-blocking)
   - Retrieval happens in BACKGROUND thread
3. Continue generation while retrieval runs
4. Inject results when ready
Total: ~1.5-1.8s (overlap reduces latency)
```

## Thread-Safe Operations

### Queue Management
- Use `queue.Queue()` for thread-safe communication
- `put_nowait()` - never blocks main thread
- `get_nowait()` - check without blocking

### Cache Strategy
- Thread-safe dict for storing retrieved facts
- Key: query token
- Value: retrieved fact
- Prevents duplicate retrievals

### Daemon Thread
- Background retrieval worker runs as daemon
- Dies when main thread dies
- No zombie threads

## Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                      Main Thread                                │
├──────────────────────────────────────────────────────────────────┤
│  Prompt → Token Generator (ollama streaming)                │
│           ↓                                               │
│        Token 1 → Gate Network → Score: 0.2                │
│           ↓                                              │
│        Queue.put_nowait(Token1)  ←────────────────┐     │
│           ↓                                       │     │
│        Check cache → Use if hit                 │     │
│           ↓                                       │     │
│        Yield token                              │     │
│           ↓                                       │     │
│        Token 2 → ... (repeat)                 │     │
│           ↓                                       │     │
│        Wait for worker to finish                │     │
│           ↓                                       │     │
│        Final Output                           │     │
└───────────────────────────────────────────────│─────────┘
                                                ↓
┌──────────────────────────────────────────────────────────────────┐
│                  Background Thread (Daemon)                       │
├──────────────────────────────────────────────────────────────────┤
│  while True:                                                   │
│    Token = Queue.get_nowait()  ←─────────────────────────┐        │
│    Result = retrieve_async(Token)                       │        │
│    Cache[Token] = Result                        │        │
│                                              │        │
│    If Queue empty and no more tokens: break    └────────┘
└──────────────────────────────────────────────────────────────────┘
```

## Performance Implications

### Parallelism Ratio
- `overlap_time / max(generation_time, retrieval_time)`
- If ratio > 0.3: good parallelization
- If ratio > 0.5: excellent

### Expected Times
- Generation: ~6-7s (same as before)
- Retrieval: ~1-2s (same operations)
- But: Run in PARALLEL
- Total: max(gen, ret) = ~6-7s (not sum)

### Optimization Strategies
1. True streaming with smaller batches
2. Cache aggressively
3. Batch similar queries
4. Use faster embedding model

## Thread Safety Checklist

- [x] Queue for communication
- [x] Daemon thread pattern
- [x] Cache with locking if needed
- [x] Non-blocking put/get
- [x] Proper cleanup on exit