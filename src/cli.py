"""
Interactive CLI interface for LitReview
"""

import os
import sys
from typing import Optional, Tuple

from .scraper.scholar_scraper import ScholarScraper
from .processor.export_manager import ExportManager
from .utils.config import Config
from .utils.logger import setup_logger

logger = setup_logger(__name__)

class LitReviewCLI:
    """
    Interactive command-line interface for LitReview
    """
    
    def __init__(self):
        logger.info("Initializing LitReview CLI...")
        try:
            self.config = self._load_config()
            logger.info(f"Configuration loaded: output_dir={self.config.output_dir}, export_format={self.config.export_format}")
            
            self.scraper = ScholarScraper(self.config)
            logger.info("Scholar scraper initialized")
            
            self.exporter = ExportManager(self.config.output_dir)
            logger.info(f"Export manager initialized with output directory: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise
        
    def run(self):
        """
        Run the interactive CLI
        """
        logger.info("Starting CLI interface")
        self._print_welcome()
        
        while True:
            try:
                query, num_results = self._get_user_input()
                if not query:
                    logger.info("No query provided, exiting...")
                    break
                    
                # Perform search
                logger.info(f"Starting search with query: '{query}', requesting {num_results} results")
                results = self.scraper.search(query, num_results)
                
                if not results:
                    logger.warning("Search completed but no results found")
                    print("No results found for your query. Please try different keywords.")
                    continue
                
                logger.info(f"Search completed successfully, found {len(results)} results")
                
                # Export results
                logger.info(f"Exporting results in {self.config.export_format} format...")
                filepath = self.exporter.export(
                    results,
                    query,
                    self.config.export_format
                )
                
                if filepath:
                    logger.info(f"Results successfully exported to: {filepath}")
                    print(f"\nResults exported to: {filepath}")
                else:
                    logger.error("Export failed")
                    print("\nFailed to export results. Please check the logs for details.")
                    
                if not self._continue_prompt():
                    logger.info("User chose to exit")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Operation cancelled by user (KeyboardInterrupt)")
                print("\nOperation cancelled by user")
                break
            except Exception as e:
                logger.error(f"An error occurred during execution: {str(e)}", exc_info=True)
                print(f"\nAn error occurred: {str(e)}")
                if not self._continue_prompt():
                    break
        
        logger.info("CLI interface terminated")
                    
    def _print_welcome(self):
        """
        Display welcome message and tool information
        """
        welcome_message = """
╔════════════════════════════════════════════╗
║           Welcome to LitReview             ║
║    Scientific Literature Search Tool       ║
╚════════════════════════════════════════════╝

This tool helps you collect scientific articles from Google Scholar.
Features:
- Search with custom keywords
- Export to Excel or CSV
- Collect metadata (DOI, abstract, citations)
- Track search progress
        """
        print(welcome_message)
        logger.info("Welcome message displayed")
        
    def _get_user_input(self) -> Tuple[Optional[str], int]:
        """
        Get search query and number of results from user
        """
        logger.info("Requesting user input")
        print("\nEnter search parameters (press Ctrl+C to exit):")
        query = input("Search query: ").strip()
        
        if not query:
            logger.info("Empty query received")
            return None, 0
            
        logger.info(f"Query received: {query}")
        
        while True:
            try:
                num_results = input(
                    f"Number of results (10-{self.config.max_results}, default=100): "
                ).strip()
                
                if not num_results:
                    logger.info("Using default number of results: 100")
                    return query, 100
                    
                num_results = int(num_results)
                if 10 <= num_results <= self.config.max_results:
                    logger.info(f"Number of results set to: {num_results}")
                    return query, num_results
                else:
                    logger.warning(f"Invalid number of results entered: {num_results}")
                    print(f"Please enter a number between 10 and {self.config.max_results}")
            except ValueError:
                logger.warning("Invalid input for number of results")
                print("Please enter a valid number")
                
    def _continue_prompt(self) -> bool:
        """
        Ask user if they want to continue
        """
        logger.info("Prompting user to continue")
        while True:
            response = input("\nWould you like to perform another search? (y/n): ").strip().lower()
            if response in ('y', 'yes'):
                logger.info("User chose to continue")
                return True
            elif response in ('n', 'no'):
                logger.info("User chose to stop")
                return False
            logger.debug(f"Invalid continue response: {response}")
            print("Please enter 'y' or 'n'")
            
    def _load_config(self) -> Config:
        """
        Load configuration from file or use defaults
        """
        config_path = os.path.join('config', 'settings.yaml')
        try:
            logger.info(f"Attempting to load configuration from: {config_path}")
            return Config.from_yaml(config_path)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            return Config.default()
