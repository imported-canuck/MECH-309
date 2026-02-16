from functools import wraps
import time
def timed(func):
    """
    Decorator to measure the wall-clock execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        print(f"[{func.__name__}] elapsed time: {t1 - t0:.6f} s")
        return result
    return wrapper