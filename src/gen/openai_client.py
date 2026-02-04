# src/gen/openai_client.py
import os
import time
from typing import Dict, List, Optional

from openai import APIError, APITimeoutError, AzureOpenAI, RateLimitError
import httpx

# Inline env functions to avoid relative import issues when loaded dynamically
ENABLE_DEBUG = os.getenv("TESTGEN_DEBUG", "0").lower() in ("1", "true", "yes")

# Timeout configuration (in seconds)
DEFAULT_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120"))  # 2 minutes default
DEFAULT_CONNECT_TIMEOUT = float(os.getenv("OPENAI_CONNECT_TIMEOUT", "30"))  # 30 seconds for connection

def get_any_env(*names: str) -> str:
    """Get environment variable from multiple possible names. Raises RuntimeError if none found."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    raise RuntimeError(f"Missing required environment variable. Tried: {', '.join(names)}")

def get_optional_env(*names: str, default: str = "") -> str:
    """Get environment variable from multiple possible names with default fallback."""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def create_client() -> AzureOpenAI:
    """Create Azure OpenAI client with comprehensive configuration and timeout."""
    try:
        # Configure timeout to prevent indefinite hangs
        timeout_config = httpx.Timeout(
            timeout=DEFAULT_TIMEOUT,           # Total timeout for the request
            connect=DEFAULT_CONNECT_TIMEOUT,   # Timeout for establishing connection
        )

        client = AzureOpenAI(
            api_key=get_any_env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"),
            azure_endpoint=get_any_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_ENDPOINT"),
            api_version=get_optional_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION", default="2023-12-01-preview"),
            timeout=timeout_config,
            max_retries=2,  # Built-in retry for transient failures
        )

        if ENABLE_DEBUG:
            print(f"Azure OpenAI client created successfully (timeout: {DEFAULT_TIMEOUT}s, connect: {DEFAULT_CONNECT_TIMEOUT}s)")

        return client

    except Exception as e:
        raise RuntimeError(f"Failed to create Azure OpenAI client: {e}")

def get_openai_client() -> AzureOpenAI:
    """Get configured Azure OpenAI client (alias for create_client for auto-fixer compatibility)."""
    return create_client()

def get_deployment_name() -> str:
    """Get the deployment name for Azure OpenAI."""
    return get_any_env("AZURE_OPENAI_DEPLOYMENT", "OPENAI_DEPLOYMENT")

def create_chat_completion(client: AzureOpenAI, deployment: str, messages: List[Dict[str, str]],
                          max_tokens: Optional[int] = None, temperature: Optional[float] = None,
                          request_timeout: Optional[float] = None) -> str:
    """
    Create chat completion with robust error handling, retry logic, and timeout.

    Args:
        client: Azure OpenAI client
        deployment: Deployment name
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate (None for model default)
        temperature: Sampling temperature (ignored for Azure OpenAI compatibility)
        request_timeout: Per-request timeout in seconds (default: 180s)

    Returns:
        Generated content as string

    Raises:
        RuntimeError: If all retry attempts fail
    """

    retry_delays = [1, 3, 6]  # Progressive backoff
    last_error = None
    effective_timeout = request_timeout or float(os.getenv("OPENAI_REQUEST_TIMEOUT", "180"))  # 3 minutes default

    # Calculate total message size for logging
    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    print(f"Starting API call: {len(messages)} messages, ~{total_chars} chars, timeout: {effective_timeout}s")

    for attempt, delay in enumerate(retry_delays + [0]):  # Extra attempt without delay
        try:
            # Apply delay for retries
            if delay > 0:
                print(f"Retry {attempt + 1}/{len(retry_delays) + 1}: waiting {delay}s before retry...")
                time.sleep(delay)

            print(f"Attempt {attempt + 1}/{len(retry_delays) + 1}: Calling Azure OpenAI API...")
            start_time = time.time()

            # Prepare request parameters - Azure OpenAI specific
            request_params = {
                "model": deployment,
                "messages": messages,
                "timeout": effective_timeout,  # Per-request timeout
            }

            # Only add max_tokens if explicitly provided
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            # NOTE: temperature parameter is intentionally omitted
            # Many Azure OpenAI deployments only support the default temperature (1.0)
            # and will return a 400 error if temperature is explicitly set

            if ENABLE_DEBUG and temperature is not None:
                print(f"Note: temperature parameter ({temperature}) ignored for Azure OpenAI compatibility")

            response = client.chat.completions.create(**request_params)

            elapsed = time.time() - start_time
            print(f"API response received in {elapsed:.1f}s")

            # Extract content from response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    print(f"Successfully generated {len(content)} characters in {elapsed:.1f}s")
                    return content
                else:
                    raise RuntimeError("Empty response content from API")
            else:
                raise RuntimeError("No choices returned from API")

        except RateLimitError as e:
            last_error = f"Rate limit exceeded: {e}"
            print(f"⚠️ Rate limit hit on attempt {attempt + 1}, will retry...")
            continue

        except APITimeoutError as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            last_error = f"API timeout after {elapsed:.1f}s: {e}"
            print(f"⚠️ Timeout on attempt {attempt + 1} after {elapsed:.1f}s, will retry...")
            continue

        except httpx.TimeoutException as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            last_error = f"HTTP timeout after {elapsed:.1f}s: {e}"
            print(f"⚠️ HTTP timeout on attempt {attempt + 1} after {elapsed:.1f}s, will retry...")
            continue

        except httpx.ConnectError as e:
            last_error = f"Connection error: {e}"
            print(f"⚠️ Connection failed on attempt {attempt + 1}: {e}")
            continue

        except APIError as e:
            last_error = f"API error: {e}"
            # Check for Azure OpenAI specific parameter errors
            if hasattr(e, 'status_code'):
                if e.status_code == 400 and "temperature" in str(e):
                    print("Azure OpenAI temperature parameter not supported, continuing without it")
                    raise RuntimeError(f"Azure OpenAI parameter error: {e}")
                # Some API errors shouldn't be retried
                elif e.status_code in [400, 401, 403]:
                    print(f"❌ Non-retryable API error (status {e.status_code}): {e}")
                    raise RuntimeError(f"Non-retryable API error: {e}")
            print(f"⚠️ API error on attempt {attempt + 1}: {e}")
            continue

        except Exception as e:
            last_error = f"Unexpected error: {type(e).__name__}: {e}"
            print(f"⚠️ Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
            continue

    # All attempts failed
    print(f"❌ All {len(retry_delays) + 1} attempts failed. Last error: {last_error}")
    raise RuntimeError(f"Chat completion failed after {len(retry_delays) + 1} attempts. Last error: {last_error}")

def validate_client_configuration() -> bool:
    """
    Validate that the OpenAI client can be configured and used.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Test client creation
        client = create_client()
        deployment = get_deployment_name()
        
        if ENABLE_DEBUG:
            print("OpenAI client configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"OpenAI client configuration validation failed: {e}")
        return False

def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for text.
    Uses simple heuristic: ~4 characters per token on average.
    """
    return len(text) // 4

def prepare_messages_for_generation(system_prompt: str, user_prompt: str, 
                                  max_total_tokens: int = 32000) -> List[Dict[str, str]]:
    """
    Prepare and validate messages for generation, ensuring they fit within token limits.
    
    Args:
        system_prompt: System message content
        user_prompt: User message content  
        max_total_tokens: Maximum total tokens for the request
    
    Returns:
        List of properly formatted messages
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Estimate token usage
    total_tokens = sum(estimate_token_count(msg["content"]) for msg in messages)
    
    if total_tokens > max_total_tokens:
        # Truncate user prompt to fit within limits
        system_tokens = estimate_token_count(system_prompt)
        available_for_user = max_total_tokens - system_tokens - 1000  # Reserve for response
        
        if available_for_user > 0:
            max_user_chars = available_for_user * 4  # Rough conversion back to characters
            if len(user_prompt) > max_user_chars:
                user_prompt = user_prompt[:max_user_chars] + "\n... (truncated for length)"
                messages[1]["content"] = user_prompt
                
                if ENABLE_DEBUG:
                    print(f"Truncated user prompt to {len(user_prompt)} characters")
        else:
            raise RuntimeError("System prompt too long - cannot fit user content")
    
    return messages

# Legacy function aliases for backward compatibility
client = create_client
deployment_name = get_deployment_name  
chat_completion_create = lambda cli, dep, msgs: create_chat_completion(cli, dep, msgs)

# Export exception classes for error handling
__all__ = ['create_client', 'get_deployment_name', 'create_chat_completion', 
           'validate_client_configuration', 'RateLimitError', 'APIError', 'APITimeoutError']