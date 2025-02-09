"""
Enhanced Google Scholar scraper with improved metadata extraction and progress tracking
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import setup_logger
from ..utils.rate_limiter import RateLimiter

logger = setup_logger(__name__)

class ScholarScraper:
    """
    Enhanced scraper for Google Scholar with progress tracking and rate limiting
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = "https://scholar.google.com/scholar"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.rate_limiter = RateLimiter(
            initial_delay=config.rate_limit_delay,
            max_delay=60.0
        )
        
    def search(self, query: str, num_results: int = 100) -> List[Dict]:
        """
        Perform a search on Google Scholar with progress tracking and rate limiting
        
        Args:
            query: Search query string
            num_results: Number of results to retrieve (in steps of 10)
            
        Returns:
            List of article dictionaries containing metadata
        """
        results = []
        num_pages = (num_results + 9) // 10
        current_page = 0
        
        with tqdm(total=num_results, desc="Collecting articles") as pbar:
            while current_page < num_pages and len(results) < num_results:
                try:
                    # Wait before making request
                    self.rate_limiter.wait()
                    
                    # Fetch page
                    page_results = self._fetch_page(query, current_page * 10)
                    
                    if page_results:
                        results.extend(page_results)
                        pbar.update(len(page_results))
                        self.rate_limiter.report_success()
                        
                        if len(page_results) < 10:  # Last page
                            break
                    else:
                        logger.warning(f"No results found on page {current_page + 1}")
                        break
                        
                    current_page += 1
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:  # Too Many Requests
                        logger.warning("Rate limit detected, increasing delay")
                        self.rate_limiter.report_failure()
                        
                        if self.rate_limiter.should_retry():
                            logger.info("Retrying after increased delay...")
                            continue
                        else:
                            logger.error("Max retries reached, stopping")
                            break
                    else:
                        logger.error(f"HTTP error occurred: {str(e)}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error fetching page {current_page + 1}: {str(e)}")
                    self.rate_limiter.report_failure()
                    
                    if not self.rate_limiter.should_retry():
                        break
                    
        return results[:num_results]
    
    def _fetch_page(self, query: str, start: int = 0) -> List[Dict]:
        """
        Fetch a single page of results from Google Scholar
        """
        params = {
            'q': query,
            'start': start,
            'hl': 'en',
            'as_sdt': '0,5'
        }
        
        response = requests.get(
            self.base_url,
            params=params,
            headers=self.headers,
            timeout=30  # Add timeout
        )
        response.raise_for_status()
        
        # Check for CAPTCHA or blocking page
        if "sorry" in response.text.lower() and "robots" in response.text.lower():
            raise requests.exceptions.HTTPError("Detected Google Scholar block page")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        
        for result in soup.select('.gs_r.gs_or.gs_scl'):
            try:
                article = self._extract_article_metadata(result)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.error(f"Error extracting article metadata: {str(e)}")
                continue
                
        return articles
    
    def _extract_article_metadata(self, result) -> Optional[Dict]:
        """
        Extract enhanced metadata from an article result
        """
        try:
            # Basic metadata
            title_elem = result.find('h3', class_='gs_rt')
            if not title_elem:
                return None
                
            title = title_elem.get_text(strip=True)
            url = title_elem.find('a')['href'] if title_elem.find('a') else None
            
            # Authors, year, and publisher
            byline = result.find('div', class_='gs_a').get_text(strip=True)
            authors, year, publisher = self._parse_byline(byline)
            
            # Citations
            citations_elem = result.select_one('.gs_fl a')
            citations = int(re.search(r'\d+', citations_elem.text).group()) if citations_elem and 'Cited by' in citations_elem.text else 0
            
            # Abstract
            abstract_elem = result.find('div', class_='gs_rs')
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else None
            
            # DOI extraction
            doi = self._extract_doi(url) if url else None
            
            return {
                'title': title,
                'url': url,
                'authors': authors,
                'first_author': authors[0] if authors else None,
                'year': year,
                'publisher': publisher,
                'citations': citations,
                'abstract': abstract,
                'doi': doi,
                'retrieved_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in metadata extraction: {str(e)}")
            return None
            
    def _parse_byline(self, byline: str) -> tuple:
        """
        Parse the byline to extract authors, year, and publisher
        """
        parts = byline.split(' - ')
        authors = [author.strip() for author in parts[0].split(',')]
        
        year = None
        publisher = None
        
        if len(parts) > 1:
            year_match = re.search(r'\b\d{4}\b', parts[1])
            year = int(year_match.group()) if year_match else None
            
        if len(parts) > 2:
            publisher = parts[2].strip()
            
        return authors, year, publisher
        
    def _extract_doi(self, url: str) -> Optional[str]:
        """
        Extract DOI from URL or article page
        """
        if not url:
            return None
            
        # Direct DOI in URL
        doi_match = re.search(r'10\.\d{4,}/[-._;()/:\w]+', url)
        if doi_match:
            return doi_match.group()
            
        try:
            # Try to find DOI in article page
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Common DOI meta tags
            doi_meta = soup.find('meta', {'name': ['citation_doi', 'dc.identifier', 'dc.Identifier', 'doi']})
            if doi_meta and 'content' in doi_meta.attrs:
                return doi_meta['content']
                
        except Exception as e:
            logger.debug(f"Could not extract DOI from article page: {str(e)}")
            
        return None
