from typing import List, Iterable
from vertexai.language_models import TextEmbeddingModel

MODEL_NAME = "text-multilingual-embedding-002"

# Be extra conservative (avoid the 20k hard cap by a big margin)
DEFAULT_BATCH_CAP = 50          # max items per request
HARD_CAP_TOKENS   = 12000       # ~12k tokens/request (well below 20k)
CHAR_CAP_PER_REQ  = 48000       # also guard by ~48k chars/request (~12k tokens, ≈ chars/4)

MODEL = TextEmbeddingModel.from_pretrained(MODEL_NAME)

def _approx_tokens(s: str) -> int:
    if not s:
        return 0
    return max(1, len(s) // 4)

def _chunk_safely(texts: List[str],
                  token_cap: int,
                  batch_cap: int,
                  char_cap: int) -> Iterable[List[str]]:
    batch: List[str] = []
    tokens_in_batch = 0
    chars_in_batch  = 0

    for t in texts:
        t_tokens = _approx_tokens(t)
        t_chars  = len(t)

        # If an individual item is huge, send it alone
        if t_tokens >= token_cap or t_chars >= char_cap:
            if batch:
                yield batch
                batch, tokens_in_batch, chars_in_batch = [], 0, 0
            yield [t]
            continue

        # Would adding this item exceed any cap?
        if (tokens_in_batch + t_tokens > token_cap) or \
           (len(batch) >= batch_cap) or \
           (chars_in_batch + t_chars > char_cap):
            if batch:
                yield batch
            batch = [t]
            tokens_in_batch = t_tokens
            chars_in_batch  = t_chars
        else:
            batch.append(t)
            tokens_in_batch += t_tokens
            chars_in_batch  += t_chars

    if batch:
        yield batch

def embed_texts(
    texts: List[str],
    batch_cap: int = DEFAULT_BATCH_CAP,
    token_cap: int = HARD_CAP_TOKENS,
    char_cap: int = CHAR_CAP_PER_REQ,
    verbose: bool = True
) -> List[List[float]]:
    """
    Safe embeddings for hundreds of short items:
      • Loads model once
      • Splits by very conservative token + char caps
      • Returns List[List[float]] vectors
    """
    if not texts:
        return []

    if verbose:
        approx_total = sum(_approx_tokens(t) for t in texts)
        print(f"[embed_texts] items={len(texts)} approx_tokens≈{approx_total} "
              f"(caps: tokens={token_cap}, chars={char_cap}, batch={batch_cap})")

    vectors: List[List[float]] = []
    done = 0
    for i, batch in enumerate(_chunk_safely(texts, token_cap, batch_cap, char_cap), start=1):
        if verbose:
            btoks = sum(_approx_tokens(t) for t in batch)
            bchars = sum(len(t) for t in batch)
            print(f"[embed_texts] → batch {i}: size={len(batch)} "
                  f"approx_tokens≈{btoks} chars={bchars}")
        embs = MODEL.get_embeddings(batch)
        vectors.extend([e.values for e in embs])
        done += len(batch)
        if verbose:
            print(f"[embed_texts]   done {done}/{len(texts)}")

    return vectors

'''
from vertexai.language_models import TextEmbeddingModel
from typing import List, Tuple

model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Takes a list of Hebrew (or multilingual) text and returns list of embeddings.
    """
    if not texts:
        return []
    # return model.get_embeddings(texts)
    embeddings = model.get_embeddings(texts)
    return [e.values for e in embeddings]
'''
