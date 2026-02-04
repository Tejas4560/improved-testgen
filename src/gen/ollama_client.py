"""
Ollama Client

Provides both embedding generation and LLM completions using Ollama's local API.
Compatible with OpenAI client interface.
"""

import requests
import os
from typing import List, Dict, Any, Optional
import time


class OllamaEmbeddingClient:
    """Client for generating embeddings using Ollama."""

    def __init__(
        self,
        host: str = None,
        model: str = None,
        vector_dim: int = None
    ):
        """
        Initialize Ollama embedding client.

        Args:
            host: Ollama host URL (defaults to OLLAMA_HOST env var)
            model: Embedding model name (defaults to OLLAMA_EMBED_MODEL env var)
            vector_dim: Vector dimension (defaults to VECTOR_DIM env var)
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "deepseek-r1:latest")
        self.vector_dim = int(vector_dim or os.getenv("VECTOR_DIM", "1024"))

        # Ensure host doesn't end with slash
        self.host = self.host.rstrip('/')

        self.embeddings_url = f"{self.host}/api/embeddings"

    def create_embedding(
        self,
        input_text: str,
        model: str = None
    ) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            input_text: Text to embed
            model: Model to use (defaults to instance model)

        Returns:
            Embedding vector as list of floats
        """
        model = model or self.model

        payload = {
            "model": model,
            "prompt": input_text
        }

        try:
            response = requests.post(
                self.embeddings_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            embedding = data.get("embedding", [])

            if not embedding:
                raise ValueError("Empty embedding returned from Ollama")

            return embedding

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Invalid Ollama response format: {e}")

    def create_embeddings_batch(
        self,
        input_texts: List[str],
        model: str = None,
        max_retries: int = 3
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            input_texts: List of texts to embed
            model: Model to use (defaults to instance model)
            max_retries: Maximum retry attempts per text

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i, text in enumerate(input_texts):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    embedding = self.create_embedding(text, model)
                    embeddings.append(embedding)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        # Fallback: return zero vector
                        print(f"Failed to embed text {i+1}/{len(input_texts)} after {max_retries} retries: {e}")
                        embeddings.append([0.0] * self.vector_dim)
                        break
                    else:
                        # Wait before retry
                        time.sleep(2 ** retry_count)  # Exponential backoff

        return embeddings


class OllamaOpenAIAdapter:
    """
    Adapter to make Ollama client compatible with OpenAI client interface.

    This allows us to use Ollama embeddings with code written for OpenAI.
    """

    def __init__(
        self,
        host: str = None,
        model: str = None,
        vector_dim: int = None
    ):
        self.client = OllamaEmbeddingClient(host, model, vector_dim)
        self.embeddings = self  # For client.embeddings.create() pattern

    def create(
        self,
        model: str,
        input: List[str]
    ):
        """
        Create embeddings (OpenAI-compatible interface).

        Args:
            model: Model name (overrides instance model)
            input: List of texts to embed

        Returns:
            Response object compatible with OpenAI format
        """
        # Handle single string input
        if isinstance(input, str):
            input = [input]

        # Generate embeddings
        embeddings = self.client.create_embeddings_batch(input, model)

        # Return OpenAI-compatible response
        return OllamaEmbeddingResponse(embeddings)


class OllamaEmbeddingResponse:
    """Response object compatible with OpenAI embedding response format."""

    def __init__(self, embeddings: List[List[float]]):
        self.data = [
            OllamaEmbeddingData(i, embedding)
            for i, embedding in enumerate(embeddings)
        ]


class OllamaEmbeddingData:
    """Data object compatible with OpenAI embedding data format."""

    def __init__(self, index: int, embedding: List[float]):
        self.index = index
        self.embedding = embedding


def get_ollama_client() -> OllamaOpenAIAdapter:
    """
    Get Ollama client with OpenAI-compatible interface (for embeddings).

    Returns:
        OllamaOpenAIAdapter instance
    """
    return OllamaOpenAIAdapter()


# ============================================================================
# LLM COMPLETION SUPPORT (for chat/reasoning)
# ============================================================================

class OllamaLLMClient:
    """Client for chat completions using Ollama."""

    def __init__(
        self,
        host: str = None,
        model: str = None
    ):
        """
        Initialize Ollama LLM client.

        Args:
            host: Ollama host URL (defaults to OLLAMA_HOST env var)
            model: Model name (defaults to OLLAMA_MODEL env var)
        """
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "deepseek-r1:latest")
        self.host = self.host.rstrip('/')
        self.chat_url = f"{self.host}/api/chat"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to instance model)
            max_tokens: Maximum tokens to generate (ignored for Ollama)
            temperature: Sampling temperature (optional)
            stream: Whether to stream response (False for now)

        Returns:
            Response dict compatible with OpenAI format
        """
        model = model or self.model

        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        # Add optional parameters
        if temperature is not None:
            payload["options"] = {"temperature": temperature}

        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=600  # 10 minutes for reasoning models
            )
            response.raise_for_status()

            data = response.json()

            # Extract message content
            message_content = data.get("message", {}).get("content", "")

            if not message_content:
                raise ValueError("Empty response from Ollama")

            # Return in OpenAI-compatible format
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": message_content
                    },
                    "finish_reason": "stop",
                    "index": 0
                }],
                "model": model,
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                }
            }

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama chat request failed: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Invalid Ollama response format: {e}")


class OllamaLLMAdapter:
    """
    Adapter to make Ollama LLM client compatible with OpenAI client interface.

    Provides client.chat.completions.create() method.
    """

    def __init__(
        self,
        host: str = None,
        model: str = None
    ):
        self.client = OllamaLLMClient(host, model)
        self.chat = self  # For client.chat.completions.create() pattern
        self.completions = self  # For client.chat.completions

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_completion_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Create chat completion (OpenAI-compatible interface).

        Args:
            model: Model name
            messages: List of message dicts
            max_completion_tokens: Max tokens (ignored for Ollama)
            temperature: Sampling temperature
            **kwargs: Other params (ignored)

        Returns:
            Response object compatible with OpenAI format
        """
        response_dict = self.client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_completion_tokens,
            temperature=temperature
        )

        return OllamaChatResponse(response_dict)


class OllamaChatResponse:
    """Response object compatible with OpenAI chat completion response."""

    def __init__(self, response_dict: Dict[str, Any]):
        self.choices = [
            OllamaChoice(choice_data)
            for choice_data in response_dict.get("choices", [])
        ]
        self.model = response_dict.get("model", "")
        self.usage = response_dict.get("usage", {})


class OllamaChoice:
    """Choice object compatible with OpenAI format."""

    def __init__(self, choice_data: Dict[str, Any]):
        self.message = OllamaMessage(choice_data.get("message", {}))
        self.finish_reason = choice_data.get("finish_reason", "stop")
        self.index = choice_data.get("index", 0)


class OllamaMessage:
    """Message object compatible with OpenAI format."""

    def __init__(self, message_data: Dict[str, str]):
        self.role = message_data.get("role", "assistant")
        self.content = message_data.get("content", "")


def get_ollama_llm_client() -> OllamaLLMAdapter:
    """
    Get Ollama LLM client with OpenAI-compatible interface.

    Returns:
        OllamaLLMAdapter instance
    """
    return OllamaLLMAdapter()


def test_ollama_connection() -> bool:
    """
    Test connection to Ollama server.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        client = get_ollama_client()

        # Try a simple embedding
        test_text = "Hello, world!"
        response = client.create(
            model=os.getenv("OLLAMA_EMBED_MODEL", "deepseek-r1:latest"),
            input=[test_text]
        )

        if response.data and len(response.data) > 0:
            print(f"Ollama connection successful!")
            print(f"Embedding dimension: {len(response.data[0].embedding)}")
            return True
        else:
            print("Ollama returned empty response")
            return False

    except Exception as e:
        print(f"Ollama connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the client
    print("Testing Ollama embedding client...")
    print(f"Host: {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    print(f"Model: {os.getenv('OLLAMA_EMBED_MODEL', 'deepseek-r1:latest')}")

    success = test_ollama_connection()

    if success:
        print("\n Client is ready to use!")
    else:
        print("\n Client setup failed. Check your Ollama configuration.")