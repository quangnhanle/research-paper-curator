import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError
from src.schemas.llm import RAGResponse

JSON_FORMAT_INSTRUCTION = (
    "Return your answer as a single JSON object with exactly these keys:\n"
    '{"answer": "...", "sources": [], "confidence": "medium", "citations": []}\n'
    "- answer: your full answer as a string\n"
    "- sources: list of PDF URLs used (may be empty)\n"
    '- confidence: "high", "medium", or "low"\n'
    "- citations: list of arXiv IDs referenced (may be empty)\n"
    "Do not include any text outside the JSON object."
)


class RAGPromptBuilder:
    """Builder class for creating RAG prompts."""

    def __init__(self):
        """Initialize the prompt builder."""
        self.prompts_dir = Path(__file__).parent / "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the text file.

        Returns:
            System prompt string
        """
        prompt_file = self.prompts_dir / "rag_system.txt"
        if not prompt_file.exists():
            # Fallback to default prompt if file doesn't exist
            return (
                "You are an AI assistant specialized in answering questions about "
                "academic papers from arXiv. Base your answer STRICTLY on the provided "
                "paper excerpts."
            )
        return prompt_file.read_text().strip()

    def create_user_content(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create the user message content with retrieved chunks and the question.

        Args:
            query: User's question
            chunks: List of retrieved chunks with metadata from OpenSearch

        Returns:
            Formatted user message string
        """
        content = "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            # Get the actual chunk text
            chunk_text = chunk.get("chunk_text", chunk.get("content", ""))
            arxiv_id = chunk.get("arxiv_id", "")

            # Only include minimal metadata - just arxiv_id for citation
            content += f"[{i}. arXiv:{arxiv_id}]\n"
            content += f"{chunk_text}\n\n"

        content += f"### Question:\n{query}\n\n"
        content += "### Answer (cite sources using [arXiv:id] format):\n"

        return content

    def create_rag_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Create a full RAG prompt (system prompt + context + question).

        Args:
            query: User's question
            chunks: List of retrieved chunks with metadata from OpenSearch

        Returns:
            Formatted prompt string
        """
        return f"{self.system_prompt}\n\n{self.create_user_content(query, chunks)}"

    def create_chat_messages(
        self, query: str, chunks: List[Dict[str, Any]], structured: bool = False
    ) -> List[Dict[str, str]]:
        """Create Chat Completions messages for a RAG request.

        Args:
            query: User's question
            chunks: List of retrieved chunks
            structured: Whether to instruct the model to answer as JSON

        Returns:
            List of message dictionaries with system and user roles
        """
        system_content = self.system_prompt
        if structured:
            system_content = f"{system_content}\n\n{JSON_FORMAT_INSTRUCTION}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": self.create_user_content(query, chunks)},
        ]


class ResponseParser:
    """Parser for LLM responses."""

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """Parse a structured response from the LLM.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with parsed response
        """
        try:
            # Try to parse as JSON and validate with Pydantic
            parsed_json = json.loads(response)
            validated_response = RAGResponse(**parsed_json)
            return validated_response.model_dump()
        except (json.JSONDecodeError, ValidationError):
            # Fallback: try to extract JSON from the response
            return ResponseParser._extract_json_fallback(response)

    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """Extract JSON from response text as fallback.

        Args:
            response: Raw response text

        Returns:
            Dictionary with extracted content or fallback
        """
        # Try to find JSON in the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate with Pydantic, using defaults for missing fields
                validated = RAGResponse(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass

        # Final fallback: wrap plain text as a structured response
        return {
            "answer": response,
            "sources": [],
            "confidence": "medium",
            "citations": [],
        }
