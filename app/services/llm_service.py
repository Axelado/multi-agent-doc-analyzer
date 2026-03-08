"""Service LLM — interface Ollama."""

from __future__ import annotations

import structlog
from langchain_ollama import ChatOllama
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = structlog.get_logger(__name__)


class LLMService:
    """Wrapper autour du LLM Ollama local."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._llm: ChatOllama | None = None

    @property
    def llm(self) -> ChatOllama:
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.ollama_model,
                base_url=self.settings.ollama_base_url,
                temperature=0.1,
                num_ctx=self.settings.ollama_num_ctx,
                num_predict=self.settings.ollama_num_predict,
            )
        return self._llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate(self, prompt: str) -> str:
        """Générer une réponse à partir d'un prompt."""
        logger.debug("llm_generate", prompt_length=len(prompt))
        response = await self.llm.ainvoke(prompt)
        result = str(self._content_to_text(response.content))
        logger.debug("llm_response", response_length=len(result))
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
    async def generate_structured(self, prompt: str, system_prompt: str = "") -> str:
        """Générer avec un system prompt."""
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))

        response = await self.llm.ainvoke(messages)
        return str(self._content_to_text(response.content))

    def _content_to_text(self, content: str | list[str | dict]) -> str:
        if isinstance(content, str):
            return content
        return "\n".join(str(part) for part in content)

    async def health_check(self) -> bool:
        """Vérifier que Ollama est accessible."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


# Singleton
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
