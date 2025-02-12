"""Module de scraping pour Google Scholar avec gestion améliorée des erreurs"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, 
    NoSuchElementException,
    WebDriverException,
    SessionNotCreatedException,
    ElementClickInterceptedException,
    StaleElementReferenceException
)
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import random
import logging
import yaml
import os
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import traceback
from ..utils.config_manager import ConfigManager
from ..utils.error_handler import (
    setup_logger,
    handle_selenium_exceptions,
    retry_on_exception,
    BrowserInitError,
    CaptchaError,
    NetworkError
)

class ScholarScraper:
    """Scraper pour Google Scholar avec gestion robuste des erreurs"""
    
    def __init__(self, config_manager: ConfigManager):
        """Initialize the scraper with configuration"""
        # Initialiser le logger en premier
        self.logger = setup_logger("scholar_scraper")
        self.driver = None
        self.wait = None
        self.base_url = "https://scholar.google.com"
        
        try:
            # Valider le gestionnaire de configuration
            if not isinstance(config_manager, ConfigManager):
                raise ValueError(f"Expected ConfigManager instance, got {type(config_manager)}")
            
            self.logger.info("Initializing scraper with configuration")
            self.config = config_manager
            self._init_driver()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scraper: {str(e)}")
            self.cleanup()
            raise

    @handle_selenium_exceptions
    @retry_on_exception(retries=3, delay=2)
    def _init_driver(self) -> None:
        """Initialize Chrome WebDriver with configuration"""
        chrome_options = Options()
        chrome_config = self.config.get_chrome_options()
        
        # Add all Chrome options from config
        for option in chrome_config.get('options', []):
            chrome_options.add_argument(option)
            
        # Add user agent
        chrome_options.add_argument(f"user-agent={chrome_config['user_agent']}")
        
        # Add experimental options
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Add preferences
        if 'prefs' in chrome_config:
            chrome_options.add_experimental_option('prefs', chrome_config['prefs'])
        
        try:
            # Use ChromeDriverManager
            driver_manager = ChromeDriverManager()
            self.logger.info("Installing ChromeDriver...")
            driver_path = driver_manager.install()
            
            # Get the actual chromedriver executable path
            if driver_path.endswith('chromedriver-win32/THIRD_PARTY_NOTICES.chromedriver'):
                driver_path = driver_path.replace('THIRD_PARTY_NOTICES.chromedriver', 'chromedriver.exe')
            
            self.logger.info(f"ChromeDriver installed at: {driver_path}")
            
            # Create service with the corrected driver path
            service = Service(executable_path=driver_path)
            
            # Initialize the driver
            self.logger.info("Initializing Chrome WebDriver...")
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Mask webdriver presence
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set a reasonable page load timeout
            self.driver.set_page_load_timeout(30)
            self.wait = WebDriverWait(self.driver, timeout=20)
            
            # Test connection with proper waits
            self.logger.info("Testing connection to Google Scholar...")
            self.driver.get(self.base_url)
            
            # Wait for page to be fully loaded
            self.wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
            time.sleep(2)  # Small delay to ensure JS execution
            
            self.logger.info("Chrome WebDriver initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chrome WebDriver: {str(e)}")
            raise BrowserInitError(f"Failed to initialize Chrome WebDriver")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"Error during driver cleanup: {str(e)}")
            finally:
                self.driver = None
                self.wait = None

    def search(self, query: str, num_results: int = 100) -> List[Dict]:
        """Perform a search on Google Scholar with comprehensive error handling"""
        results = []
        
        try:
            if not self.driver:
                self.logger.info("Reinitializing driver...")
                self._init_driver()
            
            # Construct search URL
            search_url = f"{self.base_url}/scholar?q={query}&hl=en"
            
            # Navigate to search page with proper waits
            self.logger.info(f"Navigating to: {search_url}")
            self.driver.get(search_url)
            
            # Wait for page load with multiple conditions
            try:
                # First wait for page load
                self.wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
                time.sleep(2)  # Small delay for JS
                
                # Then wait for either results or no results message
                results_present = EC.presence_of_element_located((By.ID, "gs_res_ccl_mid"))
                no_results = EC.presence_of_element_located((By.CLASS_NAME, "gs_alrt"))
                
                element = self.wait.until(lambda driver: (
                    results_present(driver) or no_results(driver)
                ))
                
                if element.get_attribute("class") == "gs_alrt":
                    self.logger.warning("No results found")
                    return results
                
            except TimeoutException:
                self.logger.error("Timeout waiting for search results")
                return results
            
            # Process results with proper delays
            processed = 0
            page = 0
            
            while processed < num_results:
                try:
                    # Extract results from current page
                    page_results = self._extract_results_from_page()
                    if not page_results:
                        break
                    
                    results.extend(page_results)
                    processed += len(page_results)
                    
                    if processed >= num_results:
                        break
                    
                    # Navigate to next page with proper delays
                    page += 1
                    next_url = f"{search_url}&start={page * 10}"
                    
                    # Random delay between pages
                    time.sleep(random.uniform(2, 4))
                    
                    self.driver.get(next_url)
                    self.wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing page {page}: {str(e)}")
                    break
            
            return results[:num_results]
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return results

    @handle_selenium_exceptions
    def _extract_results_from_page(self) -> List[Dict]:
        """Extract paper information from current page"""
        results = []
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Wait for results to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "gs_res_ccl_mid"))
                )
                
                # Get all result elements
                elements = self.driver.find_elements(By.CLASS_NAME, "gs_ri")
                
                if not elements:
                    self.logger.warning("No results found on page")
                    break
                    
                # Process each result
                for element in elements:
                    try:
                        result = self._process_result(element)
                        if result:
                            results.append(result)
                    except StaleElementReferenceException:
                        self.logger.warning("Encountered stale element, retrying...")
                        time.sleep(1)
                        continue
                        
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    self.logger.error(f"Failed to extract results after {max_retries} attempts: {str(e)}")
                    break
                time.sleep(2)  # Wait before retry
                
        return results

    @handle_selenium_exceptions
    def _process_result(self, result_element) -> Dict:
        """Process a single search result element"""
        try:
            # Wait for and extract title
            title_element = WebDriverWait(result_element, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".gs_rt a"))
            )
            title = title_element.text.strip()
            link = title_element.get_attribute("href")
            
            # Extract authors and publication info
            authors_element = result_element.find_element(By.CLASS_NAME, "gs_a")
            authors_text = authors_element.text
            authors, publication, year = self._parse_citation_info(authors_text)
            
            # Extract abstract
            abstract = ""
            try:
                abstract_element = result_element.find_element(By.CLASS_NAME, "gs_rs")
                abstract = abstract_element.text.strip()
            except NoSuchElementException:
                pass
            
            # Extract citation count
            citations = 0
            try:
                citations_element = result_element.find_element(By.PARTIAL_LINK_TEXT, "Cited by")
                citations = int(''.join(filter(str.isdigit, citations_element.text)))
            except (NoSuchElementException, ValueError):
                pass
            
            return {
                "title": title,
                "authors": authors,
                "year": year,
                "publication": publication,
                "abstract": abstract,
                "citations": citations,
                "link": link
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing result: {str(e)}")
            return None

    def _parse_citation_info(self, citation_text: str) -> Tuple[str, str, int]:
        """Parse citation info text into authors, publication, and year"""
        try:
            # Split by ' - ' to separate authors, publication, and year
            parts = citation_text.split(' - ')
            
            # Extract authors from first part
            authors = parts[0].strip() if parts else ""
            
            # Extract publication from second part if it exists
            publication = parts[1].strip() if len(parts) > 1 else ""
            
            # Try to extract year from any part of the text
            year = 0  # Default to 0 instead of None
            import re
            for part in parts:
                year_match = re.search(r'\b(19|20)\d{2}\b', part)  # Match years from 1900-2099
                if year_match:
                    year = int(year_match.group())
                    break
            
            # Log the parsed information
            self.logger.debug(f"Parsed citation info - Authors: {authors}, Publication: {publication}, Year: {year}")
            
            return authors, publication, year
            
        except Exception as e:
            self.logger.warning(f"Error parsing citation info '{citation_text}': {str(e)}")
            return "", "", 0  # Return 0 instead of None

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
