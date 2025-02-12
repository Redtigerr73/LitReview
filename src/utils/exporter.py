"""
Export manager for LitReview results
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict

class Exporter:
    """Export manager for search results"""
    
    def __init__(self, output_dir: str):
        """Initialize exporter with output directory"""
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_results(self, results: List[Dict], query: str) -> str:
        """
        Export results to Excel file
        
        Args:
            results: List of result dictionaries
            query: Search query used
            
        Returns:
            Path to exported file
        """
        try:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.xlsx"
            filepath = self.output_dir / filename
            
            # Export to Excel
            df.to_excel(filepath, index=False, engine='openpyxl')
            
            self.logger.info(f"Results exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            raise
