# src/gen/openai_client.py
import os
import time
from typing import Dict, List, Optional

from openai import APIError, APITimeoutError, AzureOpenAI, RateLimitError

# Inline env functions to avoid relative import issues when loaded dynamically
ENABLE_DEBUG = os.getenv("TESTGEN_DEBUG", "0").lower() in ("1", "true", "yes")

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
    """Create Azure OpenAI client with comprehensive configuration."""
    try:
        client = AzureOpenAI(
            api_key=get_any_env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY"),
            azure_endpoint=get_any_env("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_ENDPOINT"),
            api_version=get_optional_env("AZURE_OPENAI_API_VERSION", "OPENAI_API_VERSION", default="2023-12-01-preview"),
        )

        if ENABLE_DEBUG:
            print(f"Azure OpenAI client created successfully")

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
                          max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
    """
    Create chat completion with robust error handling and retry logic.
    
    Args:
        client: Azure OpenAI client
        deployment: Deployment name
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate (None for model default)
        temperature: Sampling temperature (ignored for Azure OpenAI compatibility)
    
    Returns:
        Generated content as string
    
    Raises:
        RuntimeError: If all retry attempts fail
    """
    
    retry_delays = [1, 3, 6]  # Progressive backoff
    last_error = None
    
    for attempt, delay in enumerate(retry_delays + [0]):  # Extra attempt without delay
        try:
            if ENABLE_DEBUG:
                print(f"Attempt {attempt + 1}: Creating chat completion...")
            
            # Apply delay for retries
            if delay > 0:
                time.sleep(delay)
            
            # Prepare request parameters - Azure OpenAI specific
            request_params = {
                "model": deployment,
                "messages": messages,
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
            
            # Extract content from response
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    if ENABLE_DEBUG:
                        print(f"Successfully generated {len(content)} characters")
                    return content
                else:
                    raise RuntimeError("Empty response content from API")
            else:
                raise RuntimeError("No choices returned from API")
                
        except RateLimitError as e:
            last_error = f"Rate limit exceeded: {e}"
            if ENABLE_DEBUG:
                print(f"Rate limit hit on attempt {attempt + 1}, retrying...")
            continue
            
        except APITimeoutError as e:
            last_error = f"API timeout: {e}"
            if ENABLE_DEBUG:
                print(f"Timeout on attempt {attempt + 1}, retrying...")
            continue
            
        except APIError as e:
            last_error = f"API error: {e}"
            # Check for Azure OpenAI specific parameter errors
            if e.status_code == 400 and "temperature" in str(e):
                if ENABLE_DEBUG:
                    print("Azure OpenAI temperature parameter not supported, continuing without it")
                # This specific error should not be retried
                raise RuntimeError(f"Azure OpenAI parameter error: {e}")
            # Some API errors shouldn't be retried
            elif e.status_code in [400, 401, 403]:
                raise RuntimeError(f"Non-retryable API error: {e}")
            if ENABLE_DEBUG:
                print(f"API error on attempt {attempt + 1}, retrying...")
            continue
            
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            if ENABLE_DEBUG:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
            continue
    
    # All attempts failed
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