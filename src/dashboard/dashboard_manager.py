"""Dashboard manager for analyzing scraped results with advanced SMS/SLR analysis"""

import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import traceback

from ..utils.pdf_exporter import AcademicPDFExporter
from ..taxonomy.taxonomy_integrator import TaxonomyIntegrator
from ..analysis.analyzer import SMSAnalyzer

class DashboardManager:
    """Manages the interactive dashboard for literature review analysis"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize dashboard manager"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.pdf_exporter = AcademicPDFExporter()
        self.taxonomy = TaxonomyIntegrator()
        self.analyzer = SMSAnalyzer(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return {}

    def create_dashboard(self, df: pd.DataFrame) -> dash.Dash:
        """Create and configure the Dash application"""
        try:
            # Initialize Dash app
            app = dash.Dash(__name__, 
                          external_stylesheets=[dbc.themes.BOOTSTRAP],
                          suppress_callback_exceptions=True)
            
            # Calculate metrics
            metrics = self._calculate_metrics(df)
            
            # Create layout
            app.layout = self._create_layout(df, metrics)
            
            # Add callbacks
            self._add_callbacks(app, df, metrics)
            
            return app
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate all necessary metrics for the dashboard"""
        try:
            # Basic metrics
            metrics = {
                'total_papers': len(df),
                'total_citations': df['citations'].sum(),
                'avg_citations': df['citations'].mean(),
                'median_citations': df['citations'].median(),
                'h_index': self._calculate_h_index(df['citations']),
                'years_range': f"{df['year'].min()}-{df['year'].max()}"
            }
            
            # Calculate additional metrics using the analyzer
            analysis_results = self.analyzer.analyze_dataset(df)
            metrics.update(analysis_results)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def _create_layout(self, df: pd.DataFrame, metrics: Dict) -> html.Div:
        """Create the dashboard layout"""
        try:
            return dbc.Container([
                # Header
                dbc.Row([
                    dbc.Col(html.H1("Analyse Systématique de la Littérature",
                                  className="text-center mb-4"))
                ]),
                
                # Summary Cards
                dbc.Row([
                    dbc.Col(self._create_metric_card("Total Articles", metrics['total_papers'])),
                    dbc.Col(self._create_metric_card("Citations Totales", metrics['total_citations'])),
                    dbc.Col(self._create_metric_card("H-index", metrics['h_index'])),
                    dbc.Col(self._create_metric_card("Impact Moyen", f"{metrics['avg_citations']:.1f}"))
                ]),
                
                # Tabs for different analyses
                dbc.Tabs([
                    # Temporal Analysis Tab
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Évolution Temporelle"),
                                dcc.Graph(figure=self._create_temporal_plot(df))
                            ])
                        ])
                    ], label="Analyse Temporelle"),
                    
                    # Topic Analysis Tab
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Distribution des Sujets"),
                                dcc.Graph(figure=self._create_topic_plot(df))
                            ])
                        ])
                    ], label="Analyse Thématique"),
                    
                    # Citation Network Tab
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Réseau de Citations"),
                                dcc.Graph(figure=self._create_network_plot(df))
                            ])
                        ])
                    ], label="Réseau de Citations"),
                    
                    # Methodology Tab
                    dbc.Tab([
                        dbc.Row([
                            dbc.Col([
                                html.H3("Méthodologies de Recherche"),
                                dcc.Graph(figure=self._create_methodology_plot(df))
                            ])
                        ])
                    ], label="Méthodologies")
                ]),
                
                # Export Section
                dbc.Row([
                    dbc.Col([
                        html.H3("Exporter les Résultats", className="text-center mt-4"),
                        html.Div([
                            dbc.Button("Télécharger PDF", 
                                     id="btn-pdf", 
                                     color="primary",
                                     className="me-2"),
                            dbc.Button("Télécharger Excel",
                                     id="btn-excel",
                                     color="secondary")
                        ], className="d-flex justify-content-center")
                    ])
                ], className="mt-4 mb-4"),
                
                # Hidden divs for storing data
                html.Div(id='pdf-download'),
                html.Div(id='excel-download'),
                
            ], fluid=True)
            
        except Exception as e:
            self.logger.error(f"Error creating layout: {str(e)}")
            return html.Div("Error loading dashboard")

    def _create_metric_card(self, title: str, value: any) -> dbc.Card:
        """Create a metric card"""
        return dbc.Card(
            dbc.CardBody([
                html.H4(title, className="card-title text-center"),
                html.H2(str(value), className="card-text text-center")
            ]),
            className="mb-4"
        )

    def _create_temporal_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create temporal analysis plot"""
        try:
            # Group by year
            yearly_data = df.groupby('year').agg({
                'title': 'count',
                'citations': 'sum'
            }).reset_index()
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces
            fig.add_trace(
                go.Scatter(x=yearly_data['year'], 
                          y=yearly_data['title'],
                          name="Publications",
                          mode='lines+markers'),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(x=yearly_data['year'],
                          y=yearly_data['citations'],
                          name="Citations",
                          mode='lines+markers'),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title="Évolution des Publications et Citations",
                xaxis_title="Année",
                template="plotly_white"
            )
            
            fig.update_yaxes(title_text="Nombre de Publications", secondary_y=False)
            fig.update_yaxes(title_text="Nombre de Citations", secondary_y=True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating temporal plot: {str(e)}")
            return go.Figure()

    def _create_topic_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create topic analysis plot"""
        try:
            # Get topic analysis from analyzer
            topic_data = self.analyzer.analyze_topics(df)
            
            # Create visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=list(topic_data.keys()),
                    y=list(topic_data.values())
                )
            ])
            
            fig.update_layout(
                title="Distribution des Sujets de Recherche",
                xaxis_title="Sujet",
                yaxis_title="Nombre d'Articles",
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating topic plot: {str(e)}")
            return go.Figure()

    def _create_network_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create citation network plot"""
        try:
            # Get network data from analyzer
            network_data = self.analyzer.analyze_citation_network(df)
            
            # Create visualization using the network data
            fig = go.Figure(data=[
                go.Scatter(
                    x=network_data['x'],
                    y=network_data['y'],
                    mode='markers+text',
                    text=network_data['labels'],
                    hoverinfo='text',
                    marker=dict(
                        size=network_data['sizes'],
                        color=network_data['colors'],
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            ])
            
            fig.update_layout(
                title="Réseau de Citations",
                showlegend=False,
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating network plot: {str(e)}")
            return go.Figure()

    def _create_methodology_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create methodology distribution plot"""
        try:
            # Get methodology data from analyzer
            method_data = self.analyzer.analyze_methodologies(df)
            
            # Create visualization
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(method_data.keys()),
                    values=list(method_data.values()),
                    hole=.3
                )
            ])
            
            fig.update_layout(
                title="Distribution des Méthodologies",
                template="plotly_white"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating methodology plot: {str(e)}")
            return go.Figure()

    def _calculate_h_index(self, citations: pd.Series) -> int:
        """Calculate h-index"""
        try:
            citations_sorted = sorted(citations, reverse=True)
            h = 0
            for i, c in enumerate(citations_sorted, 1):
                if c >= i:
                    h = i
                else:
                    break
            return h
        except Exception as e:
            self.logger.error(f"Error calculating h-index: {str(e)}")
            return 0

    def _add_callbacks(self, app: dash.Dash, df: pd.DataFrame, metrics: Dict):
        """Add callbacks for interactivity"""
        
        @app.callback(
            Output('pdf-download', 'children'),
            Input('btn-pdf', 'n_clicks'),
            prevent_initial_call=True
        )
        def generate_pdf(n_clicks):
            """Generate and download PDF report"""
            if n_clicks:
                try:
                    # Generate PDF using the exporter
                    output_path = self.pdf_exporter.generate_report(
                        df=df,
                        metrics=metrics,
                        figures={
                            'temporal': self._create_temporal_plot(df),
                            'topics': self._create_topic_plot(df),
                            'network': self._create_network_plot(df),
                            'methodology': self._create_methodology_plot(df)
                        },
                        output_dir=self.config.get('output_dir', 'data')
                    )
                    
                    return dcc.Download(
                        id='download-pdf',
                        data=output_path
                    )
                except Exception as e:
                    self.logger.error(f"Error generating PDF: {str(e)}")
                    return html.Div("Erreur lors de la génération du PDF")
        
        @app.callback(
            Output('excel-download', 'children'),
            Input('btn-excel', 'n_clicks'),
            prevent_initial_call=True
        )
        def generate_excel(n_clicks):
            """Generate and download Excel report"""
            if n_clicks:
                try:
                    # Create Excel writer
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = Path(self.config.get('output_dir', 'data')) / f'analysis_{timestamp}.xlsx'
                    
                    with pd.ExcelWriter(output_path) as writer:
                        # Write main data
                        df.to_excel(writer, sheet_name='Articles', index=False)
                        
                        # Write metrics
                        pd.DataFrame([metrics]).to_excel(writer, sheet_name='Metrics', index=False)
                        
                        # Write additional analyses
                        topic_data = self.analyzer.analyze_topics(df)
                        pd.DataFrame(topic_data.items(), columns=['Topic', 'Count']).to_excel(
                            writer, sheet_name='Topics', index=False)
                        
                        method_data = self.analyzer.analyze_methodologies(df)
                        pd.DataFrame(method_data.items(), columns=['Method', 'Count']).to_excel(
                            writer, sheet_name='Methods', index=False)
                    
                    return dcc.Download(
                        id='download-excel',
                        data=output_path
                    )
                except Exception as e:
                    self.logger.error(f"Error generating Excel: {str(e)}")
                    return html.Div("Erreur lors de la génération du fichier Excel")

    def run_dashboard(self, df: pd.DataFrame, debug: bool = False):
        """Run the dashboard"""
        try:
            app = self.create_dashboard(df)
            app.run_server(debug=debug)
        except Exception as e:
            self.logger.error(f"Error running dashboard: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
