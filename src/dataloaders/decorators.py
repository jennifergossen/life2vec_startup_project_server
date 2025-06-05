import logging
from functools import wraps
from pathlib import Path

log = logging.getLogger(__name__)

def save_parquet(path, on_validation_error="error", verify_index=False, parquet_kwargs=None):
    """Simple decorator that saves dask dataframes as parquet files"""
    if parquet_kwargs is None:
        parquet_kwargs = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute the function
            result = func(*args, **kwargs)
            
            # Save to parquet
            output_path = Path(str(path).format(self=args[0]))  # Format with self
            output_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving to parquet: {output_path}")
            result.to_parquet(output_path, **parquet_kwargs)
            
            return result
        return wrapper
    return decorator
