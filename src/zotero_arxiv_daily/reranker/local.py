from collections import Counter
import logging
import re
import warnings

import numpy as np

from .base import BaseReranker, register_reranker


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@register_reranker("local")
class LocalReranker(BaseReranker):
    def _load_encoder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)

    def _fallback_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        tokenized_texts = [_tokenize(text) for text in (s1 + s2)]
        counters = [Counter(tokens) for tokens in tokenized_texts]
        norms = [np.sqrt(sum(count * count for count in counter.values())) or 1.0 for counter in counters]

        scores = np.zeros((len(s1), len(s2)), dtype=float)
        corpus_counters = counters[len(s1):]
        corpus_norms = norms[len(s1):]

        for i, left_counter in enumerate(counters[:len(s1)]):
            left_norm = norms[i]
            left_items = list(left_counter.items())
            for j, right_counter in enumerate(corpus_counters):
                dot_product = sum(left_count * right_counter.get(token, 0) for token, left_count in left_items)
                scores[i, j] = dot_product / (left_norm * corpus_norms[j])

        return scores

    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        try:
            encoder = self._load_encoder()
            if self.config.reranker.local.encode_kwargs:
                encode_kwargs = self.config.reranker.local.encode_kwargs
            else:
                encode_kwargs = {}
            s1_feature = encoder.encode(s1, **encode_kwargs, show_progress_bar=True)
            s2_feature = encoder.encode(s2, **encode_kwargs, show_progress_bar=True)
            sim = encoder.similarity(s1_feature, s2_feature)
            return sim.numpy()
        except Exception as exc:
            logging.warning(
                "Local embedding model could not be loaded; falling back to offline token-overlap similarity: %s",
                exc,
            )
            return self._fallback_similarity_score(s1, s2)