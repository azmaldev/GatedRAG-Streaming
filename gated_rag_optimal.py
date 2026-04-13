import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import concurrent.futures


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


class GatedRAGOptimal:
    """
    Real approach from Chinese labs (Tongyi, DioR, Tok-RAG):

    1. User query arrives
    2. Gate network analyzes: Do we need external knowledge?
    3. If YES → Trigger retrieval (parallel with generation start)
    4. If NO → Generate from internal only
    5. Generate and integrate

    Result: Single retrieval call, parallel with generation
    Latency: max(retrieval, generation) instead of sum
    """

    def __init__(self, model="gemma3:1b"):
        self.model = model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.facts = FACTS
        self._build_index()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _build_index(self):
        embeddings = self.embed_model.encode(self.facts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve(self, query, k=1):
        query_embedding = self.embed_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"), k
        )
        return self.facts[indices[0][0]]

    def gate_network(self, prompt):
        """
        Gate decides UPFRONT whether retrieval is needed.
        Not token-by-token, but prompt-level decision.
        """
        prompt_lower = prompt.lower()

        question_words = ["what", "who", "when", "where", "why", "how", "which"]
        for word in question_words:
            if word in prompt_lower:
                return 0.2

        if "capital" in prompt_lower:
            return 0.2
        if "famous" in prompt_lower:
            return 0.3
        if "world cup" in prompt_lower or "won" in prompt_lower:
            return 0.3
        if "revolution" in prompt_lower or "1789" in prompt_lower:
            return 0.3

        return 0.8

    def generate_with_parallel_retrieval(self, prompt):
        """
        Optimal approach:
        - Gate decides upfront if retrieval needed
        - Start retrieval in background thread
        - Start generation in main thread
        - Both run in PARALLEL
        """
        start_time = time.time()

        gate_score = self.gate_network(prompt)

        retrieval_future = None
        retrieved_context = ""

        if gate_score < 0.6:
            keywords = []
            for word in [
                "capital",
                "famous",
                "revolution",
                "world cup",
                "france",
                "paris",
            ]:
                if word in prompt.lower():
                    keywords.append(word)

            if keywords:
                query = " ".join(keywords[:2])
                retrieval_future = self.executor.submit(self.retrieve, query)

        gen_start = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"num_predict": 200},
        )
        generation_time = time.time() - gen_start

        retrieval_time = 0.0
        if retrieval_future:
            ret_start = time.time()
            try:
                retrieved_context = retrieval_future.result(timeout=1.0)
            except:
                pass
            retrieval_time = time.time() - ret_start

        full_output = response["response"]
        if retrieved_context:
            full_output = f"Context: {retrieved_context}\n\n{response['response']}"

        total_time = time.time() - start_time

        metrics = {
            "generation_time": generation_time,
            "retrieval_time": retrieval_time,
            "gate_score": gate_score,
            "retrieval_triggered": gate_score < 0.6,
            "total_time": total_time,
        }

        return full_output, metrics

    def generate(self, prompt, use_gating=True):
        if use_gating:
            return self.generate_with_parallel_retrieval(prompt)

        start_time = time.time()
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"num_predict": 200},
        )
        total_time = time.time() - start_time

        return response["response"], {"total_time": total_time}


if __name__ == "__main__":
    print("Initializing GatedRAG-Optimal with Parallel Retrieval...")
    gated_rag = GatedRAGOptimal()

    test_prompts = [
        "What is the capital of France?",
        "The capital of France is Paris. It is known for",
        "What happened in 1789?",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)

        gate_score = gated_rag.gate_network(prompt)
        print(
            f"Gate score: {gate_score:.2f} ({'RETRIEVE' if gate_score < 0.6 else 'INTERNAL'})"
        )

        output, metrics = gated_rag.generate(prompt, use_gating=True)

        print(f"Output: {output[:100]}...")
        print(f"Metrics:")
        print(f"  Generation time: {metrics['generation_time']:.3f}s")
        print(f"  Retrieval time: {metrics['retrieval_time']:.3f}s")
        print(f"  Total time: {metrics['total_time']:.3f}s")
