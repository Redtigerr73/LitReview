"""
Rate limiter with adaptive delay and retry mechanism
"""

import time
import random
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Adaptive rate limiter with exponential backoff
    """
    
    def __init__(self, initial_delay: float = 2.0, max_delay: float = 60.0, backoff_factor: float = 2):
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.last_request_time = None
        self.consecutive_failures = 0
        self.success_streak = 0
        
    def wait(self):
        """
        Wait for the appropriate delay time with jitter
        """
        if self.last_request_time:
            # Add random jitter (±20% of current delay)
            jitter = self.current_delay * 0.2
            actual_delay = self.current_delay + random.uniform(-jitter, jitter)
            
            # Ensure minimum delay
            actual_delay = max(actual_delay, 1.0)
            
            # Calculate time to wait
            elapsed = time.time() - self.last_request_time
            if elapsed < actual_delay:
                wait_time = actual_delay - elapsed
                logger.debug(f"Waiting {wait_time:.2f} seconds before next request")
                time.sleep(wait_time)
        
        self.last_request_time = time.time()
        
    def report_success(self):
        """
        Report a successful request and potentially reduce delay
        """
        self.consecutive_failures = 0
        self.success_streak += 1
        
        # After 5 successful requests, try reducing delay
        if self.success_streak >= 5:
            self.current_delay = max(2.0, self.current_delay / 1.5)
            self.success_streak = 0
            logger.debug(f"Reduced delay to {self.current_delay:.2f} seconds")
            
    def report_failure(self):
        """
        Report a failed request and increase delay
        """
        self.consecutive_failures += 1
        self.success_streak = 0
        
        # Exponential backoff
        self.current_delay = min(
            self.max_delay,
            self.current_delay * self.backoff_factor
        )
        
        logger.warning(f"Increased delay to {self.current_delay:.2f} seconds after failure")
        
    def should_retry(self, max_retries: int = 3) -> bool:
        """
        Determine if another retry should be attempted
        """
        return self.consecutive_failures < max_retries
