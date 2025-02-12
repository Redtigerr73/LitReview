"""Export manager for saving scraped results"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ExportManager:
    """Handles exporting of scraped results to various formats"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_results(self, results: List[Dict], query: str) -> str:
        """
        Export results to Excel format with progress bar
        
        Args:
            results: List of article dictionaries
            query: Search query used
            
        Returns:
            Path to the exported Excel file
        """
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"scholar_{query.replace(' ', '_')}_{timestamp}"
        excel_path = os.path.join(self.output_dir, f"{base_name}.xlsx")
        
        # Convert to DataFrame with progress bar
        with tqdm(total=len(results), desc="Export des résultats", 
                 bar_format="{desc} |{bar}| {percentage:3.0f}%") as pbar:
            df_list = []
            for item in results:
                df_list.append(pd.Series(item))
                pbar.update(1)
            df = pd.DataFrame(df_list)
        
        # Export to Excel with formatting
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Articles')
            
            # Auto-adjust columns width
            worksheet = writer.sheets['Articles']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        # Also save as JSON for backup
        json_path = os.path.join(self.output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Exported {len(results)} results to {excel_path} and {json_path}")
        return excel_path
