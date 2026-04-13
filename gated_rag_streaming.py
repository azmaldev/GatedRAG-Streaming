import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import asyncio
import concurrent.futures
import threading
from collections import defaultdict

FACTS = [
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower is located in Paris, France.",
    "France is known for its world-renowned cuisine, including croissants, baguettes, and fine wines.",
    "The French Revolution began in 1789 and led to significant political and social changes.",
    "France has won the FIFA World Cup twice, in 1998 and 2018.",
    "Berlin is the capital and largest city of Germany.",
    "Germany is known for its engineering and automotive industry.",
    "Japan is an island nation in East Asia, known for technology and culture.",
    "Tokyo is the capital and largest city of Japan.",
    "Mount Fuji is Japan's highest mountain.",
]


class GatedRAGStreaming:
    def __init__(self, model="gemma3:1b"):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = FACTS
        self._build_index()

        self.retrieval_cache = {}
        self.pending_retrievals = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _build_index(self):
        embeddings = self.embed_model.encode(self.facts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query, k=1):
        if query in self.retrieval_cache:
            return self.retrieval_cache[query]

        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )
        result = self.facts[indices[0][0]]
        self.retrieval_cache[query] = result
        return result

    def gate_network(self, token, context):
        token_lower = token.lower()
        if token_lower in ["who", "what", "when", "where", "why", "how"]:
            return 0.2
        if token.isdigit():
            return 0.3
        if any(char.isdigit() for char in token):
            return 0.3
        if token and token[0].isupper():
            return 0.5
        if token_lower in [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        ]:
            return 0.9
        if token_lower in [
            "and",
            "or",
            "but",
            "if",
            "then",
            "in",
            "on",
            "at",
            "to",
            "for",
        ]:
            return 0.9
        return 0.9

    def _async_retrieve(self, query):
        future = self.executor.submit(self.retrieve, query)
        return future

    def generate_with_parallel_retrieval(self, prompt):
        start_time = time.time()
        retrieval_time = 0.0
        expert_routing_count = 0

        context = ""
        generated_text = ""

        pending_futures = []

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={"num_predict": 200},
        )

        for chunk in response:
            token = chunk["response"]
            generated_text += token

            clean_token = token.strip(".,;:!?\"' ")
            gate_score = self.gate_network(clean_token, context)

            if gate_score < 0.6:
                expert_routing_count += 1
                query = clean_token.strip()

                if query and query not in self.retrieval_cache:
                    future = self._async_retrieve(query)
                    pending_futures.append((query, future))
                elif query and query in self.retrieval_cache:
                    context += " " + self.retrieval_cache[query]

        ret_start = time.time()
        for query, future in pending_futures:
            try:
                result = future.result(timeout=2.0)
                context += " " + result
            except:
                pass
        retrieval_time += time.time() - ret_start

        total_time = time.time() - start_time

        metrics = {
            "generation_time": total_time - retrieval_time,
            "retrieval_time": retrieval_time,
            "expert_routing_count": expert_routing_count,
            "total_time": total_time,
        }

        return generated_text, metrics

    def generate_streaming(self, prompt, callback=None):
        start_time = time.time()
        expert_routing_count = 0
        context = ""

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={"num_predict": 200},
        )

        generated_text = ""
        for chunk in response:
            token = chunk["response"]
            generated_text += token

            if callback:
                callback(token)

            gate_score = self.gate_network(token, context)

            if gate_score < 0.6:
                expert_routing_count += 1
                query = token.strip()
                if query:
                    if query not in self.retrieval_cache:
                        future = self._async_retrieve(query)
                        try:
                            result = future.result(timeout=0.1)
                            context += " " + result
                        except:
                            pass

        total_time = time.time() - start_time
        metrics = {
            "expert_routing_count": expert_routing_count,
            "total_time": total_time,
        }

        return generated_text, metrics

    def generate(self, prompt, use_gating=True):
        if use_gating:
            return self.generate_with_parallel_retrieval(prompt)

        start_time = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={"num_predict": 200},
        )

        generated_text = ""
        for chunk in response:
            generated_text += chunk["response"]

        total_time = time.time() - start_time
        return generated_text, {"total_time": total_time}


if __name__ == "__main__":
    print("Initializing GatedRAG-Streaming with Parallel Retrieval...")
    gated_rag = GatedRAGStreaming()

    test_prompt = "The capital of France is Paris. It is known for"

    print(f"Running test with prompt: {test_prompt}")
    print("=" * 60)

    output, metrics = gated_rag.generate(test_prompt, use_gating=True)

    print(f"\nGenerated: {output}")
    print(f"\nMetrics:")
    print(f"  Generation time: {metrics['generation_time']:.3f}s")
    print(f"  Retrieval time: {metrics['retrieval_time']:.3f}s")
    print(f"  Expert routing count: {metrics['expert_routing_count']}")
    print(f"  Total time: {metrics['total_time']:.3f}s")
