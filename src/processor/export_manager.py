"""
Export manager for handling data export in various formats
"""

import os
import re
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ExportManager:
    """
    Handles exporting of scraped article data to various formats
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export(self, data: List[Dict], query: str, format: str = 'excel') -> str:
        """
        Export data to the specified format
        
        Args:
            data: List of article dictionaries
            query: Original search query
            format: Output format ('excel' or 'csv')
            
        Returns:
            Path to the exported file
        """
        if not data:
            logger.warning("No data to export")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate filename
        filename = self._generate_filename(query, len(data))
        
        # Export based on format
        if format.lower() == 'excel':
            return self._export_excel(df, filename)
        elif format.lower() == 'csv':
            return self._export_csv(df, filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def _generate_filename(self, query: str, result_count: int) -> str:
        """
        Generate filename based on query keywords and timestamp
        
        Args:
            query: Search query string
            result_count: Number of results found
            
        Returns:
            Base filename without extension
        """
        # Clean and process query
        # Remove special characters and extra spaces
        clean_query = re.sub(r'[^\w\s-]', '', query)
        # Replace multiple spaces with single underscore
        clean_query = re.sub(r'\s+', '_', clean_query.strip().lower())
        # Limit length while keeping whole words
        if len(clean_query) > 50:
            clean_query = '_'.join(clean_query[:50].split('_')[:-1])
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create filename with query, result count and timestamp
        return f"scholar_{clean_query}_{result_count}results_{timestamp}"
        
    def _export_excel(self, df: pd.DataFrame, base_name: str) -> str:
        """
        Export data to Excel format with formatting
        """
        filepath = os.path.join(self.output_dir, f"{base_name}.xlsx")
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Articles')
            
            # Auto-adjust columns width
            worksheet = writer.sheets['Articles']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
                
        logger.info(f"Exported data to Excel: {filepath}")
        return filepath
        
    def _export_csv(self, df: pd.DataFrame, base_name: str) -> str:
        """
        Export data to CSV format
        """
        filepath = os.path.join(self.output_dir, f"{base_name}.csv")
        df.to_csv(filepath, index=False)
        logger.info(f"Exported data to CSV: {filepath}")
        return filepath
