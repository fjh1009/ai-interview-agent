import logging

from openai import AsyncOpenAI, OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 25  # DashScope 单次最多 25 条

_async_client: AsyncOpenAI | None = None
_sync_client: OpenAI | None = None


def _get_base_url() -> str:
    """DashScope 兼容模式地址。新版 sk-ws- Key 必须使用业务空间专属域名。"""
    return settings.DASHSCOPE_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(
            api_key=settings.DASHSCOPE_API_KEY, base_url=_get_base_url()
        )
    return _async_client


def get_sync_client() -> OpenAI:
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(api_key=settings.DASHSCOPE_API_KEY, base_url=_get_base_url())
    return _sync_client


async def embed_text(text: str) -> list[float]:
    """单文本异步向量化"""
    try:
        resp = await get_async_client().embeddings.create(
            model=settings.KNOWLEDGE_EMBEDDING_MODEL,
            input=text,
            dimensions=settings.KNOWLEDGE_EMBEDDING_DIM,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding 失败: {e}")
        raise


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量异步向量化（自动分批，每批最多 25 条）"""
    if not texts:
        return []
    results: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        try:
            resp = await get_async_client().embeddings.create(
                model=settings.KNOWLEDGE_EMBEDDING_MODEL,
                input=batch,
                dimensions=settings.KNOWLEDGE_EMBEDDING_DIM,
            )
            results.extend([d.embedding for d in resp.data])
        except Exception as e:
            logger.error(f"批量 Embedding 第 {i // BATCH_SIZE + 1} 批失败: {e}")
            raise
    return results


def embed_text_sync(text: str) -> list[float]:
    """同步向量化（用于 Celery 任务）"""
    try:
        resp = get_sync_client().embeddings.create(
            model=settings.KNOWLEDGE_EMBEDDING_MODEL,
            input=text,
            dimensions=settings.KNOWLEDGE_EMBEDDING_DIM,
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"同步 Embedding 失败: {e}")
        raise


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """批量同步向量化（用于 Celery 任务）"""
    if not texts:
        return []
    results: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        try:
            resp = get_sync_client().embeddings.create(
                model=settings.KNOWLEDGE_EMBEDDING_MODEL,
                input=batch,
                dimensions=settings.KNOWLEDGE_EMBEDDING_DIM,
            )
            results.extend([d.embedding for d in resp.data])
        except Exception as e:
            logger.error(f"批量同步 Embedding 第 {i // BATCH_SIZE + 1} 批失败: {e}")
            raise
    return results
