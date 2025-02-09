"""
Script to analyze LitReview results with interactive file selection
"""

import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.analyzer.data_analyzer import LitReviewAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def select_file() -> str:
    """
    Open file dialog to select input file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title='Select Results File',
        filetypes=[
            ('Excel files', '*.xlsx'),
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        ]
    )
    
    return file_path

def main():
    """
    Main function to run the analysis
    """
    print("\n=== LitReview Data Analysis ===")
    print("\nThis tool will help you analyze your scholarly article data.")
    print("It will generate visualizations and a complete HTML report.")
    
    # Select input file
    print("\nPlease select your results file (Excel or CSV)...")
    file_path = select_file()
    
    if not file_path:
        print("No file selected. Exiting...")
        return
        
    try:
        # Initialize analyzer
        analyzer = LitReviewAnalyzer()
        
        # Load data
        print(f"\nLoading data from: {file_path}")
        if not analyzer.load_data(file_path):
            print("Failed to load data. Please check the file format.")
            return
            
        # Create output directory
        input_file = Path(file_path)
        output_dir = input_file.parent / 'analysis' / input_file.stem
        
        print(f"\nGenerating analysis in: {output_dir}")
        
        # Generate and export analysis
        report_path = analyzer.export_analysis(str(output_dir))
        
        if report_path:
            print(f"\nAnalysis complete! Report generated at: {report_path}")
            print("\nThe report includes:")
            print("- Summary statistics")
            print("- Year distribution visualization")
            print("- Citation analysis")
            print("- Word cloud of titles and abstracts")
            
            # Try to open the report in default browser
            try:
                import webbrowser
                webbrowser.open(f'file://{report_path}')
                print("\nOpening report in your default web browser...")
            except Exception as e:
                print(f"\nReport is ready at: {report_path}")
                
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the logs for more details.")

if __name__ == "__main__":
    main()
