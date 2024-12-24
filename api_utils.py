import tiktoken
from openai import AsyncOpenAI
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

try:
    from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS
except ImportError:
    print("Please create a config.py file with your API key and settings")
    raise

# Initialize OpenAI Async Client
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
)

# Global API tracking
api_stats = {
    "calls": 0,
    "tokens": {
        "total": 0,
        "prompt": 0,
        "completion": 0
    },
    "costs": {
        "total": 0.0
    },
    "start_time": datetime.now().isoformat()
}

# Mechanism call tracking
mechanism_stats = {
    "f_calls": 0,
    "judge_calls": 0
}

def increment_f_calls(amount: int = 1):
    """Increment the f_calls counter"""
    mechanism_stats["f_calls"] += amount

def increment_judge_calls(amount: int = 1):
    """Increment the judge_calls counter"""
    mechanism_stats["judge_calls"] += amount

def get_api_stats():
    """Get current API call statistics"""
    return api_stats.copy()

def get_mechanism_stats():
    """Get current mechanism call statistics"""
    return mechanism_stats.copy()

def reset_mechanism_stats():
    """Reset mechanism call counters"""
    global mechanism_stats
    mechanism_stats = {
        "f_calls": 0,
        "judge_calls": 0
    }

def count_tokens(text: str, model: str = 'gpt-4') -> int:
    """
    Count tokens for a given text using the appropriate tokenizer.

    Args:
        text: The text to tokenize
        model: Model name to use for tokenization

    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        return len(text) // 4  # Rough approximation as fallback

async def generate_completion_async(
    prompt: str,
    model_name: str = OPENAI_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Generate a completion using the OpenAI API with full metadata tracking.

    Args:
        prompt: The input prompt
        model_name: OpenAI model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        metadata: Optional metadata to include in response

    Returns:
        Tuple of (completion text, call metadata)
    """
    call_metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "success": False,
        "tokens": {
            "prompt": 0,
            "completion": 0,
            "total": 0
        },
        "error": None
    }

    if metadata:
        call_metadata.update(metadata)

    try:
        # Count prompt tokens
        prompt_tokens = count_tokens(prompt, model_name)
        call_metadata["tokens"]["prompt"] = prompt_tokens

        # Make API call
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract completion
        completion = response.choices[0].message.content

        # Count completion tokens
        completion_tokens = count_tokens(completion, model_name)

        # Update metadata
        call_metadata.update({
            "success": True,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens
            }
        })

        # Update global stats
        api_stats["calls"] += 1
        api_stats["tokens"]["prompt"] += prompt_tokens
        api_stats["tokens"]["completion"] += completion_tokens
        api_stats["tokens"]["total"] += prompt_tokens + completion_tokens

        return completion, call_metadata

    except Exception as e:
        error_msg = str(e)
        call_metadata["error"] = error_msg
        print(f"API call failed: {error_msg}")
        return None, call_metadata

async def generate_batch_completions_async(
    prompts: list[str],
    model_name: str = OPENAI_MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 1.0,
    batch_size: int = 5
) -> list[Tuple[Optional[str], Dict[str, Any]]]:
    """
    Generate completions for multiple prompts in batches.

    Args:
        prompts: List of prompts to process
        model_name: OpenAI model to use
        max_tokens: Maximum tokens per response
        temperature: Sampling temperature
        batch_size: Number of concurrent API calls

    Returns:
        List of (completion, metadata) tuples
    """
    from asyncio import Semaphore, gather

    # Create semaphore for rate limiting
    sem = Semaphore(batch_size)

    async def process_prompt(prompt: str) -> Tuple[Optional[str], Dict[str, Any]]:
        async with sem:
            return await generate_completion_async(
                prompt, model_name, max_tokens, temperature
            )

    # Process all prompts concurrently with rate limiting
    results = await gather(*[process_prompt(p) for p in prompts])
    return results

def get_api_stats() -> Dict[str, Any]:
    """
    Get current API usage statistics.

    Returns:
        Dictionary containing API call statistics
    """
    return {
        **api_stats,
        "end_time": datetime.now().isoformat(),
        "duration": (datetime.now() - datetime.fromisoformat(api_stats["start_time"])).total_seconds()
    }

def reset_api_stats():
    """Reset API tracking statistics."""
    global api_stats
    api_stats = {
        "calls": 0,
        "tokens": {
            "total": 0,
            "prompt": 0,
            "completion": 0
        },
        "costs": {
            "total": 0.0
        },
        "start_time": datetime.now().isoformat()
    }