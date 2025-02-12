"""
Proxy Manager for handling proxy rotation and validation
"""

import requests
import logging
import random
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class ProxyManager:
    """
    Manages a pool of proxies with automatic fetching and validation
    """
    
    def __init__(self, cache_duration: int = 30):
        self.proxies = []
        self.last_fetch = None
        self.cache_duration = cache_duration  # minutes
        
        # Initialize with known working proxies
        self.default_proxies = [
            {
                "ip": "34.175.45.228",
                "port": "80",
                "protocols": ["http"],
                "upTime": 100,
                "latency": 10,
                "speed": 1,
                "country": "FR"
            },
            {
                "ip": "51.158.154.173",
                "port": "3128",
                "protocols": ["http"],
                "upTime": 100,
                "latency": 15,
                "speed": 1,
                "country": "FR"
            },
            {
                "ip": "51.158.172.165",
                "port": "8811",
                "protocols": ["http"],
                "upTime": 100,
                "latency": 20,
                "speed": 1,
                "country": "FR"
            },
            {
                "ip": "51.15.242.202",
                "port": "3128",
                "protocols": ["http"],
                "upTime": 100,
                "latency": 25,
                "speed": 1,
                "country": "FR"
            }
        ]
        
        # Calculate scores for default proxies
        for proxy in self.default_proxies:
            proxy['score'] = (
                float(proxy['upTime']) * 0.5 +
                (100 - min(float(proxy['latency']), 100)) * 0.3 +
                (int(proxy['speed']) * 10) * 0.2
            )
        
        # Sort by score
        self.default_proxies.sort(key=lambda x: x['score'], reverse=True)
        self.proxies = self.default_proxies
        
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get a random working proxy from the pool"""
        if not self.proxies:
            self.proxies = self.default_proxies
            
        if not self.proxies:
            logger.warning("No proxies available in the pool")
            return None
            
        # Get a random proxy from top 3 best performing ones
        proxy_count = len(self.proxies)
        top_n = min(3, proxy_count)  # Take top 3 or all if less than 3
        proxy = random.choice(self.proxies[:top_n])
        
        # Convert to requests proxy format
        proxy_url = f"http://{proxy['ip']}:{proxy['port']}"
        
        logger.info(f"Using proxy: {proxy['ip']}:{proxy['port']} ({proxy['country']}, "
                   f"Score: {proxy['score']:.2f}, Latency: {proxy['latency']}ms)")
        
        return {
            'http': proxy_url,
            'https': proxy_url
        }
        
    def remove_proxy(self, proxy: Dict[str, str]):
        """Remove a failed proxy from the pool"""
        if not proxy or not isinstance(proxy, dict):
            return
            
        try:
            # Extract IP from proxy URL
            proxy_url = proxy.get('http', '') or proxy.get('https', '')
            if not proxy_url:
                return
                
            ip = proxy_url.split('://')[1].split(':')[0]
            self.proxies = [p for p in self.proxies if p['ip'] != ip]
            logger.info(f"Removed failed proxy {ip} from pool. {len(self.proxies)} proxies remaining")
            
            # Restore default proxies if pool is empty
            if not self.proxies:
                logger.info("Restoring default proxy pool")
                self.proxies = self.default_proxies
                
        except Exception as e:
            logger.error(f"Error removing proxy: {str(e)}")
            
    def _test_proxy(self, proxy: Dict) -> bool:
        """Test if a proxy works with Google Scholar"""
        try:
            proxy_url = f"http://{proxy['ip']}:{proxy['port']}"
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7'
            }
            
            response = requests.get(
                'http://scholar.google.com',  # Use HTTP first to test
                proxies=proxies,
                headers=headers,
                timeout=10,
                verify=False
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.debug(f"Proxy test failed: {str(e)}")
            return False
