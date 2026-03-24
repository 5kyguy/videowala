from __future__ import annotations

from ..vector_store import search_vectors
from .embeddings import embedding_service


def semantic_search(*, tenant_id: str, event_id: str, query: str, limit: int = 20) -> list[dict]:
    embed = embedding_service.embed_text(query, for_query=True)
    try:
        hits = search_vectors(
            tenant_id=tenant_id,
            event_id=event_id,
            query_vector=embed.vector,
            kind="multi",
            limit=limit,
        )
    except Exception:
        return []
    out = []
    for hit in hits:
        out.append(
            {
                "asset_id": hit.asset_id,
                "score": hit.score,
                "kind": hit.kind,
                "text_source": hit.text_source,
                "embedding_model": embed.model,
            }
        )
    return out

