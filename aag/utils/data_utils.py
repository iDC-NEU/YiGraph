"""Data utility functions for sampling and processing data values."""

def take_sample(value):
    """
    Extract at most two samples from a real value for LLM to infer structure.
    
    Args:
        value: The value to sample from (can be any type)
        
    Returns:
        A sampled version of the value:
        - Scalars (int, float, str, bool, None) → returned as-is
        - Lists → first 2 elements
        - Tuples → first 2 elements as tuple
        - Sets → first 2 elements as set
        - Dicts → first 2 key-value pairs
        - Other types → string representation truncated to 200 chars
    """
    # Scalars → return as-is
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    if isinstance(value, list):
        return value[:2]

    if isinstance(value, tuple):
        return tuple(value[:2])

    if isinstance(value, set):
        return set(list(value)[:2])

    if isinstance(value, dict):
        keys = list(value.keys())[:2]
        return {k: value[k] for k in keys}

    # Other types (e.g., custom objects) → convert to string, truncate length
    try:
        s = str(value)
        return s[:200]  # Avoid too long
    except:
        return "UNSUPPORTED_SAMPLE_TYPE"

