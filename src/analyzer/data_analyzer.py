"""
Data analysis and visualization module for LitReview
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LitReviewAnalyzer:
    """
    Analyzer for scholarly article data
    """
    
    def __init__(self):
        # Set up visualization style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100
        self.data = None
        self.file_path = None
        
    def load_data(self, file_path: str) -> bool:
        """
        Load data from Excel or CSV file
        """
        try:
            self.file_path = file_path
            if file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Use .xlsx or .csv")
                
            # Clean and prepare data
            self._prepare_data()
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
            
    def _prepare_data(self):
        """
        Clean and prepare data for analysis
        """
        if self.data is None:
            return
            
        # Convert year to numeric, handling errors
        self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
        
        # Convert citations to numeric, handling errors
        self.data['citations'] = pd.to_numeric(self.data['citations'], errors='coerce').fillna(0)
        
        # Convert retrieved_date to datetime
        self.data['retrieved_date'] = pd.to_datetime(self.data['retrieved_date'])
        
        # Clean text fields
        text_columns = ['title', 'abstract', 'publisher']
        for col in text_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna('').astype(str)
                
    def generate_summary(self) -> Dict:
        """
        Generate a summary of the dataset
        """
        if self.data is None:
            return {}
            
        return {
            'total_articles': len(self.data),
            'year_range': f"{int(self.data['year'].min())} - {int(self.data['year'].max())}" if not self.data['year'].isna().all() else "N/A",
            'total_citations': int(self.data['citations'].sum()),
            'avg_citations': round(self.data['citations'].mean(), 2),
            'top_publishers': self.data['publisher'].value_counts().head(5).to_dict(),
            'has_doi': (self.data['doi'].notna().sum() / len(self.data) * 100).round(2),
            'has_abstract': (self.data['abstract'].notna().sum() / len(self.data) * 100).round(2)
        }
        
    def plot_year_distribution(self, save_path: Optional[str] = None):
        """
        Plot distribution of articles by year
        """
        if self.data is None or self.data['year'].isna().all():
            return
            
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.data, x='year', bins=20)
        plt.title('Distribution of Articles by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Articles')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_citation_analysis(self, save_path: Optional[str] = None):
        """
        Create citation analysis visualizations
        """
        if self.data is None:
            return
            
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Citations distribution
        sns.histplot(data=self.data, x='citations', bins=20, ax=ax1)
        ax1.set_title('Distribution of Citations')
        ax1.set_xlabel('Number of Citations')
        
        # Top cited papers
        top_cited = self.data.nlargest(10, 'citations')
        sns.barplot(data=top_cited, y='title', x='citations', ax=ax2)
        ax2.set_title('Top 10 Most Cited Papers')
        ax2.set_xlabel('Citations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def generate_wordcloud(self, save_path: Optional[str] = None):
        """
        Generate word cloud from titles and abstracts
        """
        if self.data is None:
            return
            
        # Combine titles and abstracts
        text = ' '.join(self.data['title'].astype(str) + ' ' + self.data['abstract'].astype(str))
        
        # Create and generate word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100
        ).generate(text)
        
        # Display
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def export_analysis(self, output_dir: str) -> Optional[str]:
        """
        Generate and export analysis to the specified directory
        Returns the path to the generated report
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create plots directory
            plots_dir = output_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            # Generate visualizations
            self.plot_year_distribution(plots_dir / 'year_distribution.png')
            self.plot_citation_analysis(plots_dir / 'citation_analysis.png')
            self.generate_wordcloud(plots_dir / 'wordcloud.png')
            
            # Generate HTML report
            report_path = output_path / 'report.html'
            self._generate_report(report_path, plots_dir)
            
            # Return absolute path to the report
            return str(report_path.resolve())
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {str(e)}", exc_info=True)
            return None
            
    def _generate_report(self, report_path: Path, plots_dir: Path):
        # Generate summary
        summary = self.generate_summary()
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Literature Review Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .visualization {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Literature Review Analysis Report</h1>
            <div class="summary">
                <h2>Summary Statistics</h2>
                <ul>
                    <li>Total Articles: {summary['total_articles']}</li>
                    <li>Year Range: {summary['year_range']}</li>
                    <li>Total Citations: {summary['total_citations']}</li>
                    <li>Average Citations: {summary['avg_citations']}</li>
                    <li>Articles with DOI: {summary['has_doi']}%</li>
                    <li>Articles with Abstract: {summary['has_abstract']}%</li>
                </ul>
                
                <h3>Top Publishers:</h3>
                <ul>
                    {''.join(f'<li>{publisher}: {count}</li>' for publisher, count in summary['top_publishers'].items())}
                </ul>
            </div>
            
            <div class="visualization">
                <h2>Year Distribution</h2>
                <img src="plots/year_distribution.png" alt="Year Distribution">
            </div>
            
            <div class="visualization">
                <h2>Citation Analysis</h2>
                <img src="plots/citation_analysis.png" alt="Citation Analysis">
            </div>
            
            <div class="visualization">
                <h2>Word Cloud</h2>
                <img src="plots/wordcloud.png" alt="Word Cloud">
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
