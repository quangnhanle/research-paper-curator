import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from src.config import Settings
from src.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMException,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.services.llm.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class ExternalLLMClient:
    """Async client for OpenAI-compatible external LLM providers.

    Works with any provider exposing the Chat Completions API shape
    (OpenAI, OpenRouter, Groq, or a custom base URL).
    """

    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        """Initialize LLM client with settings."""
        self.provider = settings.llm_provider
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.default_model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.top_p = settings.llm_top_p
        self.timeout = httpx.Timeout(float(settings.llm_timeout))
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()
        # Optional transport override, used for testing with httpx.MockTransport
        self._transport = transport

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _raise_for_status(status_code: int, body_preview: str = "") -> None:
        """Map HTTP error status codes to provider-neutral exceptions.

        Error messages never include API keys or request headers.
        """
        if status_code in (401, 403):
            raise LLMAuthenticationError(
                f"LLM provider authentication failed (status {status_code}). Check LLM_API_KEY and LLM_BASE_URL."
            )
        if status_code == 429:
            raise LLMRateLimitError(f"LLM provider rate limit exceeded (status {status_code}).")
        if status_code >= 500:
            raise LLMProviderError(f"LLM provider service error (status {status_code}). {body_preview}".strip())
        if status_code >= 400:
            raise LLMException(f"LLM provider returned status {status_code}. {body_preview}".strip())

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the external LLM provider is reachable and credentials are valid.

        Uses the lightweight /models endpoint instead of a generation call.

        Returns:
            Dictionary with health status information
        """
        try:
            async with self._make_client() as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "message": f"LLM provider ({self.provider}) is reachable",
                        "model": self.default_model,
                    }
                self._raise_for_status(response.status_code)
                raise LLMException(f"LLM provider returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to LLM provider: {e}")
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"LLM provider timeout: {e}")
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"LLM health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from the provider.

        Returns:
            List of model information dictionaries
        """
        try:
            async with self._make_client() as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())

                if response.status_code == 200:
                    data = response.json()
                    return data.get("data", [])
                self._raise_for_status(response.status_code)
                raise LLMException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to LLM provider: {e}")
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"LLM provider timeout: {e}")
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Error listing models: {e}")

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "top_p": kwargs.pop("top_p", self.top_p),
            "stream": stream,
            **kwargs,
        }

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text using the Chat Completions API (non-streaming).

        Args:
            messages: Chat messages with role/content keys
            model: Model name to use; defaults to the configured model
            **kwargs: Additional generation parameters

        Returns:
            Generated text content
        """
        payload = self._build_payload(messages, model, stream=False, **kwargs)

        try:
            async with self._make_client() as client:
                logger.info(f"Sending request to LLM provider: model={payload['model']}, stream=False")
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                )

                if response.status_code != 200:
                    self._raise_for_status(response.status_code, response.text[:200])

                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as e:
                    raise LLMException(f"Unexpected response shape from LLM provider: {e}")

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to LLM provider: {e}")
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"LLM provider timeout: {e}")
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Error generating with LLM provider: {e}")

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate text with a streaming response (Server-Sent Events).

        Args:
            messages: Chat messages with role/content keys
            model: Model name to use; defaults to the configured model
            **kwargs: Additional generation parameters

        Yields:
            Chunks shaped as {"response": text_chunk, "done": False},
            ending with {"response": "", "done": True}
        """
        payload = self._build_payload(messages, model, stream=True, **kwargs)

        try:
            async with self._make_client() as client:
                logger.info(f"Starting streaming generation: model={payload['model']}")

                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        self._raise_for_status(response.status_code, body.decode(errors="replace")[:200])

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[len("data: ") :]
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse streaming chunk")
                            continue

                        try:
                            delta = chunk["choices"][0].get("delta", {})
                        except (KeyError, IndexError, TypeError):
                            continue

                        content = delta.get("content")
                        if content:
                            yield {"response": content, "done": False}

                    yield {"response": "", "done": True}

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Cannot connect to LLM provider: {e}")
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"LLM provider timeout: {e}")
        except LLMException:
            raise
        except Exception as e:
            raise LLMException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation; defaults to the configured model

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        try:
            messages = self.prompt_builder.create_chat_messages(query, chunks, structured=True)
            raw_response = await self.generate(messages=messages, model=model)

            logger.debug(f"Raw LLM response: {raw_response[:500]}")
            parsed_response = self.response_parser.parse_structured_response(raw_response)
            logger.debug(f"Parsed response: {parsed_response}")

            # Ensure sources are included if not already
            if not parsed_response.get("sources"):
                # Build PDF URLs from arxiv_ids
                sources = []
                seen_urls = set()
                for chunk in chunks:
                    arxiv_id = chunk.get("arxiv_id")
                    if arxiv_id:
                        # Build PDF URL from arxiv_id
                        arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                        if pdf_url not in seen_urls:
                            sources.append(pdf_url)
                            seen_urls.add(pdf_url)
                parsed_response["sources"] = sources

            # Add citations if not present
            if not parsed_response.get("citations"):
                # Extract unique arxiv IDs
                citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))
                parsed_response["citations"] = citations[:5]  # Limit to 5 citations

            return parsed_response

        except LLMException:
            raise
        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise LLMException(f"Failed to generate RAG answer: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate a streaming RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation; defaults to the configured model

        Yields:
            Streaming response chunks with partial answers
        """
        try:
            # Plain-text answer for streaming (no structured JSON)
            messages = self.prompt_builder.create_chat_messages(query, chunks, structured=False)

            async for chunk in self.generate_stream(messages=messages, model=model):
                yield chunk

        except LLMException:
            raise
        except Exception as e:
            logger.error(f"Error generating streaming RAG answer: {e}")
            raise LLMException(f"Failed to generate streaming RAG answer: {e}")
