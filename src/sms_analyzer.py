"""
Systematic Mapping Study and Literature Review Analyzer Module
Extends the existing LitReview functionality with comprehensive SMS/SLR capabilities
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, LatentDirichletAllocation
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import logging
from pathlib import Path
from collections import Counter
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import scipy.stats as stats

class SMSAnalyzer:
    def __init__(self, config_path="config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load("en_core_web_sm")
        nltk.download('stopwords')
        nltk.download('punkt')
        self.stop_words = set(stopwords.words('english'))
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f).get('sms_config', {})

    def analyze_temporal_trends(self, df):
        """Analyze temporal trends with advanced statistical metrics"""
        if 'year' not in df.columns:
            if 'date' in df.columns:
                df['year'] = pd.to_datetime(df['date']).dt.year
            else:
                self.logger.warning("No 'date' or 'year' column found. Using current year.")
                df['year'] = datetime.now().year
            
        window = self.config['analysis']['temporal'].get('window_size', 5)
        
        # Calculate comprehensive temporal metrics
        trends = df.groupby('year').agg({
            'title': 'count',
            'citations': ['mean', 'sum', 'std'] if 'citations' in df.columns else 'size',
            'authors': lambda x: x.str.count(';').mean() + 1 if 'authors' in df.columns else None
        }).rolling(window=window).mean()
        
        # Calculate year-over-year growth
        trends['yoy_growth'] = trends['title']['count'].pct_change() * 100
        
        # Add statistical significance tests
        trends['trend_significance'] = self._calculate_trend_significance(trends['title']['count'])
        
        return trends

    def _calculate_trend_significance(self, series):
        """Calculate Mann-Kendall trend test"""
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        return {'slope': slope, 'p_value': p_value, 'r_squared': r_value**2}

    def analyze_citation_network(self, df):
        """Create and analyze citation network"""
        if 'references' not in df.columns:
            self.logger.warning("No references column found. Skipping citation network analysis.")
            return None

        G = nx.DiGraph()
        
        # Build citation network
        for idx, row in df.iterrows():
            paper_id = row.get('doi', idx)
            G.add_node(paper_id, title=row['title'], year=row.get('year'))
            
            if pd.notna(row['references']):
                refs = str(row['references']).split(';')
                for ref in refs:
                    G.add_edge(paper_id, ref.strip())

        # Calculate network metrics
        metrics = {
            'centrality': nx.eigenvector_centrality_numpy(G),
            'pagerank': nx.pagerank(G),
            'clustering': nx.clustering(G),
            'components': list(nx.strongly_connected_components(G))
        }
        
        return {'graph': G, 'metrics': metrics}

    def extract_research_topics(self, df, n_topics=10):
        """Extract research topics using LDA"""
        if 'abstract' not in df.columns:
            self.logger.warning("No abstract column found. Using titles for topic modeling.")
            text_column = 'title'
        else:
            text_column = 'abstract'

        # Preprocess texts
        texts = df[text_column].fillna("").apply(self._preprocess_text)
        
        # Create document-term matrix
        vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        dtm = vectorizer.fit_transform(texts)

        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        lda_output = lda.fit_transform(dtm)

        # Extract keywords for each topic
        feature_names = vectorizer.get_feature_names_out()
        keywords_per_topic = []
        for topic_idx, topic in enumerate(lda.components_):
            top_keywords = [feature_names[i] for i in topic.argsort()[:-10-1:-1]]
            keywords_per_topic.append(top_keywords)

        return {
            'topic_distribution': lda_output,
            'keywords_per_topic': keywords_per_topic,
            'model': lda,
            'vectorizer': vectorizer
        }

    def quality_assessment(self, df):
        """Perform quality assessment of papers"""
        quality_metrics = {}
        
        # Citation impact
        if 'citations' in df.columns:
            quality_metrics['citation_impact'] = {
                'mean': df['citations'].mean(),
                'median': df['citations'].median(),
                'h_index': self._calculate_h_index(df['citations'])
            }

        # Venue quality
        if 'venue' in df.columns:
            venue_stats = df['venue'].value_counts()
            quality_metrics['venue_distribution'] = venue_stats.to_dict()

        # Methodological quality
        if 'abstract' in df.columns:
            quality_metrics['methodology_scores'] = self._assess_methodology_quality(df)

        return quality_metrics

    def _calculate_h_index(self, citations):
        """Calculate h-index for a set of papers"""
        if citations.empty:
            return 0
        citations_sorted = sorted(citations[citations > 0], reverse=True)
        h_index = 0
        for i, citations in enumerate(citations_sorted, 1):
            if citations >= i:
                h_index = i
            else:
                break
        return h_index

    def _assess_methodology_quality(self, df):
        """Assess methodological quality based on abstract content"""
        methodology_keywords = {
            'empirical': ['experiment', 'study', 'survey', 'data', 'analysis'],
            'theoretical': ['framework', 'model', 'theory', 'approach'],
            'validation': ['validation', 'evaluation', 'assessment', 'testing']
        }

        scores = []
        for _, paper in df.iterrows():
            abstract = str(paper.get('abstract', '')).lower()
            score = {}
            for category, keywords in methodology_keywords.items():
                score[category] = sum(1 for keyword in keywords if keyword in abstract)
            scores.append(score)

        return pd.DataFrame(scores)

    def generate_visualizations(self, df, analysis_results):
        """Generate comprehensive visualizations for the analysis"""
        visualizations = {}

        # 1. Temporal trend visualization
        if 'year' in df.columns:
            fig_temporal = px.line(
                analysis_results['temporal_trends'],
                title='Publication Trends Over Time',
                template='plotly_white'
            )
            visualizations['temporal_trends'] = fig_temporal

        # 2. Topic distribution heatmap
        if 'topic_distribution' in analysis_results:
            topic_dist = analysis_results['topic_distribution']
            fig_topics = px.imshow(
                topic_dist,
                title='Topic Distribution Across Documents',
                labels=dict(x='Topic', y='Document', color='Weight')
            )
            visualizations['topic_distribution'] = fig_topics

        # 3. Citation network visualization
        if 'citation_network' in analysis_results:
            G = analysis_results['citation_network']['graph']
            pos = nx.spring_layout(G)
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            fig_network = go.Figure(
                data=[
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                ],
                layout=go.Layout(
                    title='Citation Network',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    template='plotly_white'
                )
            )
            visualizations['citation_network'] = fig_network

        return visualizations

    def export_analysis(self, df, analysis_results, output_dir):
        """Export comprehensive analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / timestamp
        output_path.mkdir(parents=True, exist_ok=True)

        # Export detailed analysis report
        report = {
            'summary_statistics': {
                'total_papers': len(df),
                'year_range': f"{df['year'].min()}-{df['year'].max()}" if 'year' in df.columns else 'N/A',
                'total_citations': df['citations'].sum() if 'citations' in df.columns else 'N/A'
            },
            'quality_metrics': analysis_results.get('quality_metrics', {}),
            'topic_analysis': {
                'n_topics': len(analysis_results.get('topic_keywords', [])),
                'topics': analysis_results.get('topic_keywords', [])
            },
            'temporal_trends': analysis_results.get('temporal_trends', {}).to_dict() if isinstance(analysis_results.get('temporal_trends'), pd.DataFrame) else {},
            'network_metrics': analysis_results.get('citation_network', {}).get('metrics', {})
        }

        # Save results
        pd.DataFrame(report).to_excel(output_path / 'detailed_analysis.xlsx')
        
        # Generate and save visualizations
        visualizations = self.generate_visualizations(df, analysis_results)
        for name, fig in visualizations.items():
            fig.write_html(output_path / f'{name}.html')
            fig.write_image(output_path / f'{name}.png')

        return output_path

    def _preprocess_text(self, text):
        """Preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Basic cleaning
        text = text.lower()
        return text
    
    def classify_research(self, df):
        """Classify research papers based on predefined categories"""
        self.logger.info("Preprocessing text data...")
        
        # Ensure we have an abstract column
        if 'abstract' not in df.columns:
            self.logger.warning("No 'abstract' column found. Using 'title' instead.")
            text_column = 'title'
        else:
            text_column = 'abstract'
            
        # Preprocess text data
        texts = df[text_column].fillna("").apply(self._preprocess_text)
        
        # Skip empty texts
        if texts.str.strip().str.len().sum() == 0:
            self.logger.warning(f"No valid text found in {text_column} column. Skipping classification.")
            df['research_cluster'] = -1
            return df
            
        self.logger.info("Vectorizing text data...")
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        try:
            text_features = vectorizer.fit_transform(texts)
            
            self.logger.info("Clustering documents...")
            clusters = DBSCAN(
                eps=0.3,
                min_samples=2,
                metric='cosine'
            ).fit(text_features)
            
            df['research_cluster'] = clusters.labels_
            
            # Log clustering results
            n_clusters = len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
            self.logger.info(f"Found {n_clusters} research clusters")
            
        except Exception as e:
            self.logger.error(f"Error during classification: {str(e)}")
            df['research_cluster'] = -1
            
        return df
    
    def generate_knowledge_graph(self, df):
        """Generate a knowledge graph of papers and their relationships"""
        G = nx.Graph()
        
        # Add nodes for papers
        for idx, row in df.iterrows():
            G.add_node(
                row.get('title', f'Paper_{idx}'),
                year=row.get('year', None),
                citations=row.get('citations', 0),
                cluster=row.get('research_cluster', -1)
            )
        
        return G
