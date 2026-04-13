import sys
import time

sys.path.insert(0, "..")
from gated_rag_streaming import GatedRAGStreaming


class BenchmarkRunner:
    def __init__(self):
        self.gated_rag = GatedRAGStreaming(model="gemma3:1b")
        self.test_queries = [
            "The capital of France is Paris. It is known for",
            "The French Revolution occurred in 1789. It was a",
            "France won the World Cup in 1998. The team was",
        ]

    def run_benchmarks(self):
        results = []

        print("=" * 60)
        print("GatedRAG-Streaming Benchmark")
        print("=" * 60)

        for i, query in enumerate(self.test_queries):
            print(f"\nTest {i + 1}: {query}")
            print("-" * 40)

            output, metrics = self.gated_rag.generate(query, use_gating=True)

            print(f"Output: {output[:100]}...")
            print(f"Metrics:")
            print(f"  Generation time: {metrics['generation_time']:.3f}s")
            print(f"  Retrieval time: {metrics['retrieval_time']:.3f}s")
            print(f"  Expert routing count: {metrics['expert_routing_count']}")
            print(f"  Total time: {metrics['total_time']:.3f}s")

            results.append(
                {
                    "query": query,
                    "output": output,
                    "metrics": metrics,
                }
            )

        return results

    def print_summary(self, results):
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"{'Query':<40} | {'Total Time':<10} | {'Status'}")
        print("-" * 60)

        for result in results:
            total = result["metrics"]["total_time"]
            status = "PASS" if total < 2.0 else "FAIL"
            print(f"{result['query'][:38]:<40} | {total:>8.3f}s | {status}")

        avg_time = sum(r["metrics"]["total_time"] for r in results) / len(results)
        print("-" * 60)
        print(f"Average time: {avg_time:.3f}s")
        print(f"Target: < 2.0s (parallel retrieval)")
        print(f"Status: {'PASS' if avg_time < 2.0 else 'FAIL'}")


if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.run_benchmarks()
    runner.print_summary(results)
