"""Rate limiter for controlling request frequency"""

import time
import logging
import random

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter with exponential backoff"""
    
    def __init__(self, initial_delay: float = 5.0, max_delay: float = 30.0, backoff_factor: float = 1.5):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = initial_delay
        self.last_request_time = 0
        
    def wait(self):
        """Wait for the required delay"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.current_delay:
            time.sleep(self.current_delay - time_since_last)
        self.last_request_time = time.time()
        
    def get_delay(self) -> float:
        """Get current delay value"""
        return self.current_delay
        
    def report_success(self):
        """Report successful request, reduce delay"""
        self.current_delay = max(
            self.initial_delay,
            self.current_delay / self.backoff_factor
        )
        
    def report_failure(self):
        """Report failed request, increase delay"""
        self.current_delay = min(
            self.max_delay,
            self.current_delay * self.backoff_factor
        )
        
    def should_retry(self, max_retries: int = 3) -> bool:
        """
        Determine if another retry should be attempted
        """
        return True  # Always allow retries, as success_streak is not tracked
