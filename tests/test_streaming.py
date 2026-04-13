import sys
import time
import threading
import queue

sys.path.insert(0, "..")
from gated_rag_streaming import GatedRAGStreaming


def test_gate_scoring():
    print("Testing gate network scoring...")
    gated_rag = GatedRAGStreaming()

    assert gated_rag.gate_network("who", "") < 0.6
    assert gated_rag.gate_network("2024", "") < 0.6
    assert gated_rag.gate_network("Paris", "") == 0.5
    assert gated_rag.gate_network("the", "") == 0.9
    assert gated_rag.gate_network("and", "") == 0.9

    print("✓ Gate scoring passed")


def test_retrieval():
    print("Testing retrieval...")
    gated_rag = GatedRAGStreaming()

    result = gated_rag.retrieve("capital of France")
    assert result is not None
    assert "Paris" in result

    print("✓ Retrieval passed")


def test_cache():
    print("Testing cache...")
    gated_rag = GatedRAGStreaming()

    query = "test query"
    result1 = gated_rag.retrieve(query)

    cached = gated_rag.retrieval_cache.get(query)
    assert cached is not None

    print("✓ Cache passed")


def test_retrieval_worker():
    print("Testing retrieval worker background...")
    gated_rag = GatedRAGStreaming()

    def worker():
        for i in range(5):
            gated_rag.retrieval_queue.put(f"token_{i}")
            time.sleep(0.01)
        gated_rag.retrieval_queue.put(None)

    worker_thread = threading.Thread(target=worker)
    worker_thread.start()
    worker_thread.join()

    print("✓ Retrieval worker passed")


def test_streaming_generation():
    print("Testing streaming generation...")
    gated_rag = GatedRAGStreaming()

    output, metrics = gated_rag.generate("Hello world", use_gating=False)
    assert output is not None
    assert "total_time" in metrics

    print("✓ Streaming generation passed")


if __name__ == "__main__":
    test_gate_scoring()
    test_retrieval()
    test_cache()
    test_retrieval_worker()
    test_streaming_generation()
    print("\n✓ All tests passed!")
