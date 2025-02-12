"""PDF export functionality for academic analysis reports"""

import os
from typing import Dict, List
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.io as pio
import tempfile
from pathlib import Path
import logging

class AcademicPDFExporter:
    """Class for generating academic-style PDF reports"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def generate_report(self, df: pd.DataFrame, metrics: Dict, figures: Dict, output_dir: str) -> str:
        """Generate a comprehensive academic PDF report"""
        try:
            # Create PDF object
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(190, 10, 'Systematic Literature Review Analysis Report', 0, 1, 'C')
            
            # Date
            pdf.set_font('Arial', '', 10)
            pdf.cell(190, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
            
            # Executive Summary
            self._add_section_title(pdf, 'Executive Summary')
            self._add_metrics_summary(pdf, metrics)
            
            # Temporal Analysis
            self._add_section_title(pdf, 'Temporal Analysis')
            if 'temporal_analysis' in figures:
                self._add_figure(pdf, figures['temporal_analysis'], 'Publication and Citation Trends')
            
            # Topic Analysis
            self._add_section_title(pdf, 'Topic Analysis')
            if 'topic_analysis' in figures:
                self._add_figure(pdf, figures['topic_analysis'], 'Research Topics Distribution')
            
            # Citation Network
            self._add_section_title(pdf, 'Citation Network Analysis')
            if 'citation_network' in figures:
                self._add_figure(pdf, figures['citation_network'], 'Citation Network Visualization')
            
            # Methodology Distribution
            self._add_section_title(pdf, 'Research Methodology Analysis')
            if 'methodology_dist' in figures:
                self._add_figure(pdf, figures['methodology_dist'], 'Distribution of Research Methodologies')
            
            # Top Authors and Venues
            self._add_section_title(pdf, 'Top Contributors')
            self._add_top_contributors(pdf, metrics)
            
            # Quality Assessment
            self._add_section_title(pdf, 'Quality Assessment')
            self._add_quality_metrics(pdf, metrics)
            
            # Save the report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f'slr_report_{timestamp}.pdf')
            pdf.output(output_path)
            
            self.logger.info(f"PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
            raise
    
    def _add_section_title(self, pdf: FPDF, title: str):
        """Add a section title to the PDF"""
        pdf.set_font('Arial', 'B', 14)
        pdf.ln(10)
        pdf.cell(190, 10, title, 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.ln(5)
    
    def _add_metrics_summary(self, pdf: FPDF, metrics: Dict):
        """Add summary metrics to the PDF"""
        summary_text = [
            f"Total Papers: {metrics['total_papers']}",
            f"Total Citations: {metrics['total_citations']}",
            f"H-index: {metrics['h_index']}",
            f"Average Impact: {metrics['avg_impact']:.2f}",
            f"Time Period: {metrics['years_range']}",
            f"Growth Rate: {metrics['growth_rate']:.1f}%"
        ]
        
        for text in summary_text:
            pdf.cell(190, 8, text, 0, 1, 'L')
    
    def _add_figure(self, pdf: FPDF, fig, caption: str):
        """Add a figure to the PDF"""
        try:
            # Create temporary file for the figure
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                # Save figure as PNG
                if isinstance(fig, plt.Figure):
                    fig.savefig(tmp.name, bbox_inches='tight', dpi=300)
                else:  # Plotly figure
                    pio.write_image(fig, tmp.name, format='png', width=1000, height=600)
                
                # Add to PDF
                pdf.image(tmp.name, x=10, w=190)
                
                # Add caption
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(190, 5, caption, 0, 1, 'C')
                
                # Clean up
                os.unlink(tmp.name)
                
        except Exception as e:
            self.logger.warning(f"Could not add figure: {str(e)}")
            pdf.cell(190, 5, "Figure could not be generated", 0, 1, 'C')
    
    def _add_top_contributors(self, pdf: FPDF, metrics: Dict):
        """Add top authors and venues information"""
        # Top Authors
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, 'Top Authors by Impact:', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        
        for author, impact in metrics['top_authors'].items():
            pdf.cell(190, 6, f"{author}: {impact} citations", 0, 1, 'L')
        
        pdf.ln(5)
        
        # Top Venues
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 8, 'Top Publication Venues:', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        
        for venue, count in metrics['top_venues'].items():
            pdf.cell(190, 6, f"{venue}: {count} papers", 0, 1, 'L')
    
    def _add_quality_metrics(self, pdf: FPDF, metrics: Dict):
        """Add quality assessment metrics"""
        stats = metrics['citation_stats']
        quality_text = [
            f"Mean Citations per Paper: {stats['mean']:.2f}",
            f"Median Citations: {stats['median']:.1f}",
            f"Citation Standard Deviation: {stats['std']:.2f}",
            f"Citation Distribution Skewness: {stats['skew']:.2f}",
            f"Papers with 10+ citations (i10-index): {metrics['i10_index']}"
        ]
        
        for text in quality_text:
            pdf.cell(190, 8, text, 0, 1, 'L')
