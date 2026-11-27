import os

import numpy as np

GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

_client = None


def get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


class GeminiEmbedder:
    batch_size = 100

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        rows = []
        for i in range(0, len(texts), self.batch_size):
            result = get_client().models.embed_content(
                model=EMBED_MODEL,
                contents=texts[i : i + self.batch_size],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            rows.extend(e.values for e in result.embeddings)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1, norms)

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def generate(system: str, prompt: str) -> str:
    from google.genai import types

    response = get_client().models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            max_output_tokens=600,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""