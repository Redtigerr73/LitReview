"""
Taxonomy Integration Module
Connects the SRE taxonomy system with the existing LitReview framework
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from typing import Dict, List
import logging
import numpy as np
from .sre_taxonomy import SRETaxonomy

class TaxonomyIntegrator:
    def __init__(self, taxonomy: SRETaxonomy = None):
        self.taxonomy = taxonomy if taxonomy else SRETaxonomy()
        self.logger = logging.getLogger(__name__)
        
    def process_papers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process papers through the taxonomy system"""
        self.logger.info(f"Starting taxonomy classification for {len(df)} papers...")
        
        # Initialize taxonomy columns
        for category in self.taxonomy.categories:
            df[f'taxonomy_{category}'] = None
            
        # Classify each paper
        classifications = []
        for idx, row in df.iterrows():
            title = str(row.get('title', '')).strip()
            abstract = str(row.get('abstract', '')).strip()
            
            if not title and not abstract:
                self.logger.warning(f"Paper at index {idx} has no title or abstract")
                continue
                
            self.logger.debug(f"Processing paper: {title[:50]}...")
            classification = self.taxonomy.classify_paper(title, abstract)
            classifications.append(classification)
            
            # Store classifications in DataFrame
            for category, subcats in classification.items():
                df.at[idx, f'taxonomy_{category}'] = ','.join(subcats) if subcats else None
                
        # Log classification statistics
        for category in self.taxonomy.categories:
            col = f'taxonomy_{category}'
            non_null = df[col].notna().sum()
            self.logger.info(f"{category}: {non_null} papers classified")
        
        # Generate and store statistics
        stats = self.taxonomy.get_taxonomy_stats(classifications)
        self._export_taxonomy_stats(stats)
        
        return df
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Generate taxonomy-specific visualizations"""
        self.logger.info("Generating visualizations...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Category Distribution Sunburst
        self.create_taxonomy_sunburst(df, output_dir)
        
        # 2. Temporal Evolution of Categories
        if 'year' in df.columns:
            self._create_temporal_category_evolution(df, output_dir)
        else:
            self.logger.warning("No 'year' column found for temporal evolution")
        
        # 3. Method-Application Correlation
        self._create_category_correlation(df, output_dir)
        
    def create_taxonomy_sunburst(self, df: pd.DataFrame, output_dir: str):
        """Create interactive sunburst diagram of taxonomy distribution"""
        self.logger.info("Creating taxonomy sunburst visualization...")
        
        # Prepare data for visualization
        all_data = []
        for category in self.taxonomy.categories:
            col = f'taxonomy_{category}'
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            # Get non-null values and split multi-value entries
            values = df[col].dropna().str.split(',', expand=True).stack()
            if len(values) == 0:
                self.logger.warning(f"No data found for category {category}")
                continue
                
            counts = values.value_counts()
            self.logger.info(f"Category {category} distribution: {dict(counts)}")
            
            # Create hierarchy
            labels = [category] + list(counts.index)
            parents = [''] + [category] * len(counts)
            values = [counts.sum()] + list(counts.values)
            
            all_data.extend(zip(labels, parents, values))
        
        if not all_data:
            self.logger.warning("No data available for sunburst visualization")
            return
            
        # Create visualization
        labels, parents, values = zip(*all_data)
        fig = go.Figure(go.Sunburst(
            ids=labels,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="SRE Research Taxonomy Distribution",
            width=800,
            height=800
        )
        
        output_file = f"{output_dir}/taxonomy_sunburst.html"
        fig.write_html(output_file)
        self.logger.info(f"Saved sunburst visualization to {output_file}")
        
    def _create_temporal_category_evolution(self, df: pd.DataFrame, output_dir: str):
        """Create timeline of category evolution"""
        self.logger.info("Creating temporal evolution visualization...")
        
        if 'year' not in df.columns:
            self.logger.error("No 'year' column found for temporal evolution")
            return
            
        for category in self.taxonomy.categories:
            col = f'taxonomy_{category}'
            if col not in df.columns:
                continue
                
            # Prepare data
            temporal_data = []
            for year in sorted(df['year'].unique()):
                year_data = df[df['year'] == year][col].dropna()
                if len(year_data) == 0:
                    continue
                    
                # Split multiple categories and count
                subcats = year_data.str.split(',', expand=True).stack()
                counts = subcats.value_counts()
                
                for subcat, count in counts.items():
                    temporal_data.append({
                        'year': year,
                        'subcategory': subcat,
                        'count': count
                    })
            
            if not temporal_data:
                self.logger.warning(f"No temporal data for category {category}")
                continue
                
            # Create DataFrame for plotting
            temp_df = pd.DataFrame(temporal_data)
            
            # Create visualization
            fig = px.line(temp_df, 
                         x='year', 
                         y='count', 
                         color='subcategory',
                         title=f"Evolution of {category.title()} Over Time",
                         labels={'count': 'Number of Papers', 'year': 'Year'})
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Number of Papers",
                legend_title="Subcategories",
                hovermode='x unified'
            )
            
            output_file = f"{output_dir}/temporal_{category}.html"
            fig.write_html(output_file)
            self.logger.info(f"Saved temporal evolution for {category} to {output_file}")
                
    def _create_category_correlation(self, df: pd.DataFrame, output_dir: str):
        """Create correlation matrix between methods and applications"""
        self.logger.info("Creating category correlation visualization...")
        
        methods_col = 'taxonomy_methods'
        apps_col = 'taxonomy_applications'
        
        if methods_col not in df.columns or apps_col not in df.columns:
            self.logger.error("Required columns not found for correlation matrix")
            return
            
        # Create correlation matrix
        methods_data = df[methods_col].dropna().str.split(',', expand=True).stack()
        apps_data = df[apps_col].dropna().str.split(',', expand=True).stack()
        
        if len(methods_data) == 0 or len(apps_data) == 0:
            self.logger.warning("No data available for correlation matrix")
            return
            
        correlation_data = pd.crosstab(methods_data, apps_data)
        
        # Create visualization
        fig = px.imshow(correlation_data,
                       title="Methods-Applications Correlation",
                       labels=dict(x="Applications", y="Methods", color="Count"),
                       aspect="auto")
        
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title="Applications",
            yaxis_title="Methods"
        )
        
        output_file = f"{output_dir}/category_correlation.html"
        fig.write_html(output_file)
        self.logger.info(f"Saved correlation matrix to {output_file}")
    
    def _export_taxonomy_stats(self, stats: Dict):
        """Export taxonomy statistics to JSON"""
        output_dir = "data/taxonomy"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_file = f"{output_dir}/taxonomy_stats.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Exported taxonomy statistics to {output_file}")
