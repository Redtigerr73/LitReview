"""
Main entry point for LitReview - Unified workflow for scraping and analysis
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Optional, Tuple
import webbrowser

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.scraper.scholar_scraper import ScholarScraper
from src.processor.export_manager import ExportManager
from src.analyzer.data_analyzer import LitReviewAnalyzer
from src.utils.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class LitReviewCLI:
    """
    Interactive CLI for the complete LitReview workflow
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.scraper = ScholarScraper(self.config)
        self.exporter = ExportManager(self.config.output_dir)
        self.analyzer = LitReviewAnalyzer()
        
    def run(self):
        """
        Run the complete workflow
        """
        self._print_welcome()
        
        while True:
            try:
                # Step 1: Choose operation mode
                mode = self._get_operation_mode()
                if not mode:
                    break
                    
                if mode == "scrape":
                    # Step 2a: Get search parameters and scrape
                    file_path = self._run_scraping_workflow()
                elif mode == "analyze":
                    # Step 2b: Select existing file
                    file_path = self._select_existing_file()
                else:
                    logger.error(f"Invalid mode: {mode}")
                    continue
                    
                if not file_path:
                    continue
                    
                # Step 3: Ask if user wants to analyze the data
                if mode == "scrape" and not self._should_analyze():
                    continue
                    
                # Step 4: Run analysis
                self._run_analysis(file_path)
                
                # Step 5: Ask if user wants to continue
                if not self._continue_prompt():
                    break
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                break
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}", exc_info=True)
                print(f"\nAn error occurred: {str(e)}")
                if not self._continue_prompt():
                    break
                    
    def _print_welcome(self):
        """
        Display welcome message
        """
        welcome_message = """
╔════════════════════════════════════════════╗
║           Welcome to LitReview             ║
║    Scientific Literature Analysis Tool     ║
╚════════════════════════════════════════════╝

This tool helps you:
1. Collect scientific articles from Google Scholar
2. Export data to Excel/CSV
3. Generate visualizations and analysis reports
        """
        print(welcome_message)
        
    def _get_operation_mode(self) -> Optional[str]:
        """
        Get the operation mode from user
        """
        print("\nChoose operation mode:")
        print("1. Scrape new articles from Google Scholar")
        print("2. Analyze existing results file")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                return "scrape"
            elif choice == "2":
                return "analyze"
            elif choice == "3":
                return None
            else:
                print("Please enter a valid choice (1-3)")
                
    def _run_scraping_workflow(self) -> Optional[str]:
        """
        Run the scraping workflow
        """
        # Get search parameters
        query, num_results = self._get_search_parameters()
        if not query:
            return None
            
        print(f"\nSearching for: {query}")
        print(f"Requesting {num_results} results...")
        
        # Perform search
        results = self.scraper.search(query, num_results)
        
        if not results:
            print("No results found")
            return None
            
        # Export results
        filepath = self.exporter.export(
            results,
            query,
            self.config.export_format
        )
        
        if filepath:
            print(f"\nResults exported to: {filepath}")
            return filepath
            
        return None
        
    def _get_search_parameters(self) -> Tuple[Optional[str], int]:
        """
        Get search parameters from user
        """
        print("\nEnter search parameters (press Ctrl+C to exit):")
        query = input("Search query: ").strip()
        
        if not query:
            return None, 0
            
        while True:
            try:
                num_results = input(
                    f"Number of results (10-{self.config.max_results}, default=100): "
                ).strip()
                
                if not num_results:
                    return query, 100
                    
                num_results = int(num_results)
                if 10 <= num_results <= self.config.max_results:
                    return query, num_results
                else:
                    print(f"Please enter a number between 10 and {self.config.max_results}")
            except ValueError:
                print("Please enter a valid number")
                
    def _select_existing_file(self) -> Optional[str]:
        """
        Open file dialog to select existing results file
        """
        print("\nPlease select your results file (Excel or CSV)...")
        
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title='Select Results File',
            filetypes=[
                ('Excel files', '*.xlsx'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ]
        )
        
        if not file_path:
            print("No file selected")
            return None
            
        return file_path
        
    def _should_analyze(self) -> bool:
        """
        Ask if user wants to analyze the data
        """
        while True:
            response = input("\nWould you like to analyze the results now? (y/n): ").strip().lower()
            if response in ('y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            print("Please enter 'y' or 'n'")
            
    def _run_analysis(self, file_path: str):
        """
        Run analysis on the data file
        """
        print(f"\nAnalyzing data from: {file_path}")
        
        # Load data
        if not self.analyzer.load_data(file_path):
            print("Failed to load data. Please check the file format.")
            return
            
        # Create output directory
        input_file = Path(file_path)
        output_dir = input_file.parent / 'analysis' / input_file.stem
        
        print(f"\nGenerating analysis in: {output_dir}")
        
        # Generate and export analysis
        report_path = self.analyzer.export_analysis(str(output_dir))
        
        if report_path:
            print(f"\nAnalysis complete! Report generated at: {report_path}")
            print("\nThe report includes:")
            print("- Summary statistics")
            print("- Year distribution visualization")
            print("- Citation analysis")
            print("- Word cloud of titles and abstracts")
            
            # Try to open the report in default browser
            try:
                # Convert the path to a proper file URL
                file_url = Path(report_path).absolute().as_uri()
                webbrowser.open(file_url)
                print("\nOpening report in your default web browser...")
            except Exception as e:
                print(f"\nCouldn't open browser automatically. Report is ready at: {report_path}")
                print(f"You can open it manually in your web browser.")
                
    def _continue_prompt(self) -> bool:
        """
        Ask if user wants to continue
        """
        while True:
            response = input("\nWould you like to perform another operation? (y/n): ").strip().lower()
            if response in ('y', 'yes'):
                return True
            elif response in ('n', 'no'):
                return False
            print("Please enter 'y' or 'n'")
            
    def _load_config(self) -> Config:
        """
        Load configuration from file or use defaults
        """
        config_path = os.path.join('config', 'settings.yaml')
        try:
            return Config.from_yaml(config_path)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            logger.info("Using default configuration")
            return Config.default()

def main():
    """
    Main entry point
    """
    try:
        cli = LitReviewCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the logs for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
