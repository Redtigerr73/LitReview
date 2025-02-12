"""Module d'analyse scientifique pour Systematic Literature Review"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
from io import BytesIO
import base64
import pdfkit
from datetime import datetime
import os
import json
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Initialize NLTK and download required packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ScientificAnalyzer:
    """Analyseur scientifique pour SLR"""
    
    def __init__(self, results):
        """Initialise l'analyseur avec les résultats"""
        self.df = pd.DataFrame(results)
        self.lemmatizer = WordNetLemmatizer()
        self.prepare_data()
        
    def prepare_data(self):
        """Prépare les données pour l'analyse"""
        try:
            # Nettoyer les données
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
            self.df['citations'] = pd.to_numeric(self.df['citations'], errors='coerce')
            
            # Remplacer les valeurs manquantes
            self.df['year'] = self.df['year'].fillna(0)
            self.df['citations'] = self.df['citations'].fillna(0)
            self.df['title'] = self.df['title'].fillna('Sans titre')
            self.df['abstract'] = self.df['abstract'].fillna('')
            self.df['authors'] = self.df['authors'].fillna('Non disponible')
            
            # Prétraitement du texte
            def process_row(row):
                title = str(row.get('title', ''))
                abstract = str(row.get('abstract', ''))
                return self.preprocess_text(title + ' ' + abstract)
            
            self.df['processed_text'] = self.df.apply(process_row, axis=1)
            
            # Extraire les entités nommées et mots-clés
            self.df['named_entities'] = self.df['processed_text'].apply(self.extract_named_entities)
            self.df['keywords'] = self.df['processed_text'].apply(self.extract_keywords)
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {str(e)}")
            raise
        
    def analyze_slr(self):
        """Effectue une analyse complète pour SLR"""
        try:
            # Basic statistics
            basic_stats = {
                'total_papers': len(self.df),
                'year_range': f"{int(self.df['year'].min())} - {int(self.df['year'].max())}" if len(self.df) > 0 else "N/A",
                'total_citations': int(self.df['citations'].sum()),
                'avg_citations': round(self.df['citations'].mean(), 2) if len(self.df) > 0 else 0
            }

            # Temporal analysis
            temporal_analysis = self.analyze_temporal_trends()

            # Citation analysis
            citation_analysis = {
                'citation_ranges': self.get_citation_ranges(),
                'top_cited_papers': self.get_top_cited_papers(10)
            }

            # Author analysis
            author_analysis = self.analyze_authors()

            # Topic analysis
            topic_analysis = self.analyze_topics()

            # Methodology analysis
            methodology_analysis = self.analyze_methodology()

            # Quality assessment
            quality_assessment = self.assess_quality()

            # Research gaps
            research_gaps = self.identify_research_gaps()

            return {
                'basic_stats': basic_stats,
                'temporal_analysis': temporal_analysis,
                'citation_analysis': citation_analysis,
                'author_analysis': author_analysis,
                'topic_analysis': topic_analysis,
                'methodology_analysis': methodology_analysis,
                'quality_assessment': quality_assessment,
                'research_gaps': research_gaps
            }
        except Exception as e:
            logger.error(f"Error in analyze_slr: {str(e)}")
            return {
                'basic_stats': {
                    'total_papers': 0,
                    'year_range': 'N/A',
                    'total_citations': 0,
                    'avg_citations': 0
                },
                'temporal_analysis': {
                    'yearly_papers': [],
                    'yearly_citations': [],
                    'growth_rates': []
                },
                'citation_analysis': {
                    'citation_ranges': {},
                    'top_cited_papers': []
                },
                'author_analysis': {
                    'total_authors': 0,
                    'avg_authors_per_paper': 0,
                    'top_authors': [],
                    'collaboration_stats': {
                        'avg_authors_per_paper': 0,
                        'max_authors_on_paper': 0,
                        'single_author_papers': 0,
                        'multi_author_papers': 0
                    }
                },
                'topic_analysis': {
                    'top_keywords': {},
                    'clusters': [],
                    'topic_evolution': []
                },
                'methodology_analysis': {
                    'methodology_distribution': {},
                    'primary_studies': 0,
                    'secondary_studies': 0
                },
                'quality_assessment': {
                    'quality_metrics': {
                        'has_abstract': 0,
                        'has_citations': 0,
                        'recent_papers': 0,
                        'high_impact': 0,
                        'avg_citations_per_paper': 0,
                        'h_index': 0,
                        'i10_index': 0
                    },
                    'quality_distribution': {
                        'high': 0,
                        'medium': 0,
                        'low': 0
                    },
                    'summary': {
                        'overall_quality_score': 0
                    }
                },
                'research_gaps': {
                    'emerging_topics': [],
                    'underexplored_areas': [],
                    'future_directions': []
                }
            }

    def get_basic_stats(self):
        """Calcule les statistiques de base"""
        valid_years = self.df[self.df['year'] > 0]['year']
        valid_citations = self.df[self.df['citations'] >= 0]['citations']
        
        return {
            'total_papers': len(self.df),
            'year_range': f"{int(valid_years.min()) if len(valid_years) > 0 else 'N/A'} - {int(valid_years.max()) if len(valid_years) > 0 else 'N/A'}",
            'total_citations': int(valid_citations.sum()),
            'avg_citations': round(float(valid_citations.mean()), 2),
            'median_citations': int(valid_citations.median()),
            'h_index': self.calculate_h_index(),
            'i10_index': self.calculate_i10_index(),
            'papers_with_abstract': int(self.df['abstract'].notna().sum()),
            'papers_without_abstract': int(self.df['abstract'].isna().sum())
        }
        
    def analyze_temporal_trends(self):
        """Analyse les tendances temporelles"""
        yearly_counts = self.df[self.df['year'] > 0]['year'].value_counts().sort_index()
        yearly_citations = self.df[self.df['year'] > 0].groupby('year')['citations'].sum()
        
        # Calculer la croissance annuelle
        growth_rates = []
        for i in range(1, len(yearly_counts)):
            prev_count = yearly_counts.iloc[i-1]
            curr_count = yearly_counts.iloc[i]
            growth_rate = ((curr_count - prev_count) / prev_count) * 100 if prev_count > 0 else 0
            growth_rates.append({
                'year': yearly_counts.index[i],
                'growth_rate': round(growth_rate, 2)
            })
        
        return {
            'yearly_papers': yearly_counts.to_dict(),
            'yearly_citations': yearly_citations.to_dict(),
            'growth_rates': growth_rates,
            'peak_year': int(yearly_counts.idxmax()) if len(yearly_counts) > 0 else None,
            'peak_count': int(yearly_counts.max()) if len(yearly_counts) > 0 else 0
        }
        
    def analyze_citations(self):
        """Analyse approfondie des citations"""
        citations = self.df['citations']
        return {
            'distribution': {
                'min': int(citations.min()),
                'max': int(citations.max()),
                'mean': round(float(citations.mean()), 2),
                'median': int(citations.median()),
                'std': round(float(citations.std()), 2)
            },
            'percentiles': {
                '25th': int(citations.quantile(0.25)),
                '50th': int(citations.quantile(0.50)),
                '75th': int(citations.quantile(0.75)),
                '90th': int(citations.quantile(0.90))
            },
            'top_cited_papers': self.get_top_cited_papers(10),
            'citation_ranges': self.get_citation_ranges()
        }
        
    def analyze_authors(self):
        """Analyse des auteurs et collaborations"""
        try:
            all_authors = []
            for authors in self.df['authors']:
                if isinstance(authors, str):
                    all_authors.extend([a.strip() for a in authors.split(',')])
            
            author_counts = Counter(all_authors)
            top_authors = []
            for author, count in author_counts.most_common(10):
                top_authors.append({
                    'name': author,
                    'papers': count,
                    'total_citations': self.get_author_citations(author)
                })
            
            papers_with_authors = self.df[self.df['authors'].notna()]
            authors_per_paper = []
            for authors in papers_with_authors['authors']:
                if isinstance(authors, str):
                    authors_per_paper.append(len(authors.split(',')))
                else:
                    authors_per_paper.append(0)
            
            return {
                'total_authors': len(set(all_authors)),
                'avg_authors_per_paper': round(len(all_authors) / len(self.df), 2) if len(self.df) > 0 else 0,
                'top_authors': top_authors,
                'collaboration_stats': {
                    'avg_authors_per_paper': round(np.mean(authors_per_paper), 2) if authors_per_paper else 0,
                    'max_authors_on_paper': int(np.max(authors_per_paper)) if authors_per_paper else 0,
                    'single_author_papers': int(sum(1 for x in authors_per_paper if x == 1)),
                    'multi_author_papers': int(sum(1 for x in authors_per_paper if x > 1))
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_authors: {str(e)}")
            return {
                'total_authors': 0,
                'avg_authors_per_paper': 0,
                'top_authors': [],
                'collaboration_stats': {
                    'avg_authors_per_paper': 0,
                    'max_authors_on_paper': 0,
                    'single_author_papers': 0,
                    'multi_author_papers': 0
                }
            }

    def analyze_topics(self):
        """Analyse des sujets et tendances thématiques"""
        try:
            # Extraire les mots-clés les plus fréquents
            all_keywords = []
            for keywords in self.df['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            keyword_freq = Counter(all_keywords)
            
            # Analyser les abstracts avec TF-IDF
            if len(self.df) > 0:
                tfidf = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = tfidf.fit_transform(self.df['processed_text'])
                
                # Clustering des documents
                n_clusters = min(5, len(self.df))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                
                # Identifier les termes caractéristiques de chaque cluster
                cluster_terms = []
                feature_names = tfidf.get_feature_names_out()
                for i in range(n_clusters):
                    center_vector = kmeans.cluster_centers_[i]
                    top_terms_idx = center_vector.argsort()[-5:][::-1]
                    top_terms = [feature_names[idx] for idx in top_terms_idx]
                    cluster_terms.append({
                        'cluster': i,
                        'terms': top_terms,
                        'size': int((clusters == i).sum())
                    })
            else:
                cluster_terms = []
            
            return {
                'top_keywords': dict(keyword_freq.most_common(20)),
                'clusters': cluster_terms,
                'topic_evolution': self.analyze_topic_evolution()
            }
        except Exception as e:
            logger.error(f"Error in analyze_topics: {str(e)}")
            return {
                'top_keywords': {},
                'clusters': [],
                'topic_evolution': []
            }

    def analyze_topic_evolution(self):
        """Analyse l'évolution des sujets au fil du temps"""
        try:
            if len(self.df) == 0:
                return []
                
            # Grouper par année
            yearly_topics = defaultdict(Counter)
            for _, row in self.df.iterrows():
                year = int(row['year']) if pd.notna(row['year']) else 0
                if year >= 2015:  # Limiter aux dernières années
                    keywords = row.get('keywords', [])
                    if isinstance(keywords, list):
                        yearly_topics[year].update(keywords)
            
            # Suivre l'évolution des top sujets
            evolution = []
            for year in sorted(yearly_topics.keys()):
                top_topics = yearly_topics[year].most_common(5)
                evolution.append({
                    'year': year,
                    'topics': [{
                        'topic': topic,
                        'count': count
                    } for topic, count in top_topics]
                })
                
            return evolution
        except Exception as e:
            logger.error(f"Error in analyze_topic_evolution: {str(e)}")
            return []

    def analyze_methodology(self):
        """Analyse des méthodologies de recherche"""
        try:
            # Mots-clés indiquant différentes méthodologies
            methodology_keywords = {
                'empirical': ['empirical', 'experiment', 'study', 'observation'],
                'theoretical': ['theoretical', 'theory', 'framework', 'model'],
                'review': ['review', 'survey', 'literature', 'systematic'],
                'case_study': ['case study', 'case-study', 'industrial'],
                'simulation': ['simulation', 'simulator', 'monte carlo']
            }
            
            methodology_counts = defaultdict(int)
            for _, row in self.df.iterrows():
                text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
                for method, keywords in methodology_keywords.items():
                    if any(keyword in text for keyword in keywords):
                        methodology_counts[method] += 1
            
            return {
                'methodology_distribution': dict(methodology_counts),
                'primary_studies': len(self.df) - methodology_counts['review'],
                'secondary_studies': methodology_counts['review']
            }
        except Exception as e:
            logger.error(f"Error in analyze_methodology: {str(e)}")
            return {
                'methodology_distribution': {},
                'primary_studies': 0,
                'secondary_studies': 0
            }

    def assess_quality(self):
        """Évaluation de la qualité des études"""
        try:
            quality_metrics = {
                'has_abstract': int(self.df['abstract'].notna().sum()),
                'has_citations': int((self.df['citations'] > 0).sum()),
                'recent_papers': int((self.df['year'] >= 2020).sum())
            }

            # Calculate citation quartiles for papers with citations
            citations = self.df['citations'].dropna()
            if len(citations) > 0:
                q75 = citations.quantile(0.75)
                q25 = citations.quantile(0.25)
                quality_metrics['high_impact'] = int((self.df['citations'] >= q75).sum())
            else:
                q75 = 0
                q25 = 0
                quality_metrics['high_impact'] = 0

            quality_distribution = {
                'high': int((self.df['citations'] >= q75).sum()) if q75 > 0 else 0,
                'medium': int((self.df['citations'] >= q25).sum() - (self.df['citations'] >= q75).sum()) if q25 > 0 else 0,
                'low': int((self.df['citations'] < q25).sum()) if q25 > 0 else len(self.df)
            }

            # Additional quality metrics
            quality_metrics['avg_citations_per_paper'] = round(self.df['citations'].mean(), 2)
            quality_metrics['h_index'] = self.calculate_h_index()
            quality_metrics['i10_index'] = self.calculate_i10_index()

            return {
                'quality_metrics': quality_metrics,
                'quality_distribution': quality_distribution,
                'summary': {
                    'overall_quality_score': round((quality_metrics['has_abstract'] + 
                                                  quality_metrics['has_citations'] + 
                                                  quality_metrics['recent_papers'] + 
                                                  quality_metrics['high_impact']) / (4 * len(self.df)) * 100, 2)
                    if len(self.df) > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error in assess_quality: {str(e)}")
            return {
                'quality_metrics': {
                    'has_abstract': 0,
                    'has_citations': 0,
                    'recent_papers': 0,
                    'high_impact': 0,
                    'avg_citations_per_paper': 0,
                    'h_index': 0,
                    'i10_index': 0
                },
                'quality_distribution': {
                    'high': 0,
                    'medium': 0,
                    'low': 0
                },
                'summary': {
                    'overall_quality_score': 0
                }
            }

    def identify_research_gaps(self):
        """Identification des lacunes de recherche"""
        try:
            recent_years = self.df[self.df['year'] >= 2020]
            emerging_topics = self.analyze_emerging_topics(recent_years)
            
            return {
                'emerging_topics': emerging_topics,
                'underexplored_areas': self.find_underexplored_areas(),
                'future_directions': self.suggest_future_directions()
            }
        except Exception as e:
            logger.error(f"Error in identify_research_gaps: {str(e)}")
            return {
                'emerging_topics': [],
                'underexplored_areas': [],
                'future_directions': []
            }

    def analyze_emerging_topics(self, recent_df):
        """Analyse des sujets émergents"""
        try:
            if len(recent_df) == 0:
                return []
                
            tfidf = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(recent_df['processed_text'])
            
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            emerging_topics = []
            for idx in scores.argsort()[-10:][::-1]:
                emerging_topics.append({
                    'topic': feature_names[idx],
                    'score': round(float(scores[idx]), 4)
                })
                
            return emerging_topics
        except Exception as e:
            logger.error(f"Error in analyze_emerging_topics: {str(e)}")
            return []

    def find_underexplored_areas(self):
        """Identifie les domaines peu explorés"""
        try:
            # Analyser les zones avec peu de publications mais citations élevées
            topic_impact = defaultdict(lambda: {'papers': 0, 'citations': 0})
            
            for _, row in self.df.iterrows():
                keywords = row.get('keywords', [])
                if isinstance(keywords, list):
                    for keyword in keywords:
                        topic_impact[keyword]['papers'] += 1
                        topic_impact[keyword]['citations'] += row.get('citations', 0)
            
            underexplored = []
            for topic, stats in topic_impact.items():
                if stats['papers'] < 3 and stats['citations'] > 0:  # Critères arbitraires
                    underexplored.append({
                        'topic': topic,
                        'papers': stats['papers'],
                        'avg_citations': round(stats['citations'] / stats['papers'], 2)
                    })
            
            return sorted(underexplored, key=lambda x: x['avg_citations'], reverse=True)[:5]
        except Exception as e:
            logger.error(f"Error in find_underexplored_areas: {str(e)}")
            return []

    def get_top_cited_papers(self, n=10):
        """Retourne les n articles les plus cités"""
        top_papers = self.df.nlargest(n, 'citations')
        return [{
            'title': row['title'],
            'authors': row['authors'],
            'year': int(row['year']) if pd.notna(row['year']) else None,
            'citations': int(row['citations'])
        } for _, row in top_papers.iterrows()]
        
    def get_citation_ranges(self):
        """Analyse la distribution des citations par plage"""
        ranges = [
            (0, 0, 'Non cité'),
            (1, 10, '1-10 citations'),
            (11, 50, '11-50 citations'),
            (51, 100, '51-100 citations'),
            (101, float('inf'), '100+ citations')
        ]
        
        distribution = {}
        for start, end, label in ranges:
            count = len(self.df[(self.df['citations'] >= start) & (self.df['citations'] <= end)])
            distribution[label] = int(count)
            
        return distribution
        
    def get_author_citations(self, author):
        """Calcule le total des citations pour un auteur"""
        total_citations = 0
        for _, row in self.df.iterrows():
            if isinstance(row['authors'], str) and author in row['authors']:
                total_citations += row['citations']
        return int(total_citations)
        
    def analyze_collaborations(self, all_authors):
        """Analyse les collaborations entre auteurs"""
        # Calculer le nombre moyen d'auteurs par article
        papers_with_authors = self.df[self.df['authors'].notna()]
        authors_per_paper = [len(authors.split(',')) for authors in papers_with_authors['authors']]
        
        return {
            'avg_authors_per_paper': round(np.mean(authors_per_paper), 2),
            'max_authors_on_paper': int(np.max(authors_per_paper)),
            'single_author_papers': int(sum(1 for x in authors_per_paper if x == 1)),
            'multi_author_papers': int(sum(1 for x in authors_per_paper if x > 1))
        }
        
    def analyze_topic_evolution(self):
        """Analyse l'évolution des sujets au fil du temps"""
        if len(self.df) == 0:
            return []
            
        # Grouper par année
        yearly_topics = defaultdict(Counter)
        for _, row in self.df.iterrows():
            year = int(row['year']) if pd.notna(row['year']) else 0
            if year >= 2015:  # Limiter aux 5 dernières années
                if isinstance(row.get('keywords', []), list):
                    yearly_topics[year].update(row['keywords'])
        
        # Suivre l'évolution des top sujets
        evolution = []
        for year in sorted(yearly_topics.keys()):
            top_topics = yearly_topics[year].most_common(5)
            evolution.append({
                'year': year,
                'topics': [{
                    'topic': topic,
                    'count': count
                } for topic, count in top_topics]
            })
            
        return evolution
        
    def suggest_future_directions(self):
        """Suggère des directions futures de recherche"""
        # Basé sur l'analyse des tendances et des lacunes
        recent_growth = self.analyze_temporal_trends()['growth_rates'][-3:]
        emerging_topics = self.analyze_emerging_topics(self.df[self.df['year'] >= 2020])
        
        suggestions = []
        if recent_growth and emerging_topics:
            for topic in emerging_topics[:3]:
                suggestions.append({
                    'topic': topic['topic'],
                    'rationale': 'Sujet émergent avec forte croissance récente',
                    'potential_impact': round(topic['score'] * 100, 2)
                })
        
        return suggestions
        
    def create_dashboard(self):
        """Crée un dashboard académique complet"""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        app.layout = dbc.Container([
            # En-tête
            html.Div([
                html.H1("Systematic Literature Review Analysis", className="text-center my-4"),
                html.P("Analyse systématique de la littérature scientifique", className="text-center text-muted mb-4")
            ]),
            
            # Métriques principales
            dbc.Row([
                dbc.Col(self.create_metric_card("Articles", self.metrics['basic_stats']['total_papers']), width=3),
                dbc.Col(self.create_metric_card("Citations", self.metrics['basic_stats']['total_citations']), width=3),
                dbc.Col(self.create_metric_card("H-index", self.metrics['basic_stats']['h_index']), width=3),
                dbc.Col(self.create_metric_card("I10-index", self.metrics['basic_stats']['i10_index']), width=3)
            ], className="mb-4"),
            
            # Onglets d'analyse
            dbc.Tabs([
                # Vue d'ensemble
                dbc.Tab([
                    html.Div([
                        html.H3("Résumé de l'analyse", className="mt-4 mb-3"),
                        self.create_summary_section()
                    ])
                ], label="Vue d'ensemble"),
                
                # Analyse temporelle
                dbc.Tab([
                    html.Div([
                        html.H3("Évolution temporelle", className="mt-4 mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=self.create_time_series()), width=6),
                            dbc.Col(dcc.Graph(figure=self.create_citation_trends()), width=6)
                        ])
                    ])
                ], label="Analyse temporelle"),
                
                # Analyse des auteurs
                dbc.Tab([
                    html.Div([
                        html.H3("Analyse des auteurs", className="mt-4 mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=self.create_author_network()), width=12),
                            dbc.Col(dcc.Graph(figure=self.create_author_impact()), width=12)
                        ])
                    ])
                ], label="Analyse des auteurs"),
                
                # Analyse thématique
                dbc.Tab([
                    html.Div([
                        html.H3("Analyse thématique", className="mt-4 mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=self.create_topic_evolution()), width=12),
                            dbc.Col(dcc.Graph(figure=self.create_keyword_network()), width=12)
                        ])
                    ])
                ], label="Analyse thématique"),
                
                # Cartographie systématique
                dbc.Tab([
                    html.Div([
                        html.H3("Cartographie systématique", className="mt-4 mb-3"),
                        dbc.Row([
                            dbc.Col(dcc.Graph(figure=self.create_systematic_map()), width=12),
                            dbc.Col(dcc.Graph(figure=self.create_research_clusters()), width=12)
                        ])
                    ])
                ], label="Cartographie"),
                
                # Données brutes
                dbc.Tab([
                    html.Div([
                        html.H3("Données brutes", className="mt-4 mb-3"),
                        self.create_data_table()
                    ])
                ], label="Données")
            ]),
            
            # Export
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.H3("Exporter l'analyse", className="mt-4 mb-3"),
                    dbc.Button(
                        "Télécharger le rapport complet (PDF)",
                        id="btn-export-pdf",
                        color="primary",
                        className="me-2"
                    ),
                    dbc.Button(
                        "Exporter les données (Excel)",
                        id="btn-export-excel",
                        color="secondary",
                        className="me-2"
                    ),
                    dcc.Download(id="download-pdf"),
                    dcc.Download(id="download-excel")
                ])
            ], className="mt-4")
            
        ], fluid=True)
        
        @app.callback(
            dash.Output("download-pdf", "data"),
            dash.Input("btn-export-pdf", "n_clicks"),
            prevent_initial_call=True
        )
        def export_pdf(n_clicks):
            if n_clicks:
                return self.generate_pdf_report()
                
        @app.callback(
            dash.Output("download-excel", "data"),
            dash.Input("btn-export-excel", "n_clicks"),
            prevent_initial_call=True
        )
        def export_excel(n_clicks):
            if n_clicks:
                return dcc.send_data_frame(
                    self.df.to_excel,
                    "systematic_review_data.xlsx",
                    sheet_name="Articles"
                )
        
        return app
        
    def create_metric_card(self, title, value):
        """Crée une carte de métrique"""
        return dbc.Card([
            dbc.CardBody([
                html.H4(title, className="card-title text-center"),
                html.H2(str(value), className="text-center")
            ])
        ])
        
    def create_summary_section(self):
        """Crée la section de résumé"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Statistiques globales"),
                    html.Ul([
                        html.Li(f"Nombre total d'articles : {self.metrics['basic_stats']['total_papers']}"),
                        html.Li(f"Total des citations : {self.metrics['basic_stats']['total_citations']}"),
                        html.Li(f"Moyenne des citations : {self.metrics['basic_stats']['avg_citations']}"),
                        html.Li(f"Période couverte : {self.metrics['basic_stats']['years_range']}")
                    ])
                ], width=6),
                dbc.Col([
                    html.H4("Top 5 des auteurs"),
                    html.Ul([
                        html.Li(f"{author}: {count} articles") 
                        for author, count in list(self.metrics['author_metrics']['top_authors'].items())[:5]
                    ])
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Principaux sujets de recherche"),
                    html.Ul([
                        html.Li(f"Topic {topic['id']}: {', '.join(topic['terms'][:5])}") 
                        for topic in self.metrics['topic_metrics']['topics']
                    ])
                ])
            ])
        ])
        
    def create_data_table(self):
        """Crée un tableau de données interactif"""
        return dash_table.DataTable(
            data=self.df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in self.df.columns],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_current=0,
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            }
        )
        
    def generate_pdf_report(self):
        """Génère un rapport PDF académique complet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"systematic_review_{timestamp}.pdf"
        
        # Créer le contenu HTML avec style académique
        html_content = f"""
        <html>
        <head>
            <title>Systematic Literature Review Report</title>
            <style>
                body {{
                    font-family: "Times New Roman", Times, serif;
                    line-height: 1.6;
                    margin: 40px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                h1 {{
                    font-size: 24px;
                    text-align: center;
                }}
                h2 {{
                    font-size: 20px;
                    margin-top: 30px;
                }}
                h3 {{
                    font-size: 18px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f5f5f5;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .abstract {{
                    font-style: italic;
                    margin: 20px 40px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Systematic Literature Review Report</h1>
                <p>Generated on {datetime.now().strftime("%d %B %Y, %H:%M")}</p>
            </div>
            
            <div class="section">
                <h2>1. Executive Summary</h2>
                <p>This systematic literature review analyzed {self.metrics['basic_stats']['total_papers']} 
                scientific articles published between {self.metrics['basic_stats']['years_range']}. 
                The corpus has accumulated {self.metrics['basic_stats']['total_citations']} citations 
                with an h-index of {self.metrics['basic_stats']['h_index']}.</p>
            </div>
            
            <div class="section">
                <h2>2. Research Metrics</h2>
                <h3>2.1 Citation Analysis</h3>
                <ul>
                    <li>Total Citations: {self.metrics['basic_stats']['total_citations']}</li>
                    <li>Average Citations: {self.metrics['basic_stats']['avg_citations']}</li>
                    <li>H-index: {self.metrics['basic_stats']['h_index']}</li>
                    <li>i10-index: {self.metrics['basic_stats']['i10_index']}</li>
                </ul>
                
                <h3>2.2 Top Authors</h3>
                <table>
                    <tr>
                        <th>Author</th>
                        <th>Publication Count</th>
                    </tr>
                    {''.join(f"<tr><td>{author}</td><td>{count}</td></tr>"
                            for author, count in list(self.metrics['author_metrics']['top_authors'].items())[:10])}
                </table>
            </div>
            
            <div class="section">
                <h2>3. Research Topics</h2>
                {''.join(f"<h3>Topic {i+1}</h3><p>{', '.join(topic['terms'])}</p>"
                        for i, topic in enumerate(self.metrics['topic_metrics']['topics']))}
            </div>
            
            <div class="section">
                <h2>4. Detailed Article Analysis</h2>
                <table>
                    <tr>
                        <th>Title</th>
                        <th>Authors</th>
                        <th>Year</th>
                        <th>Citations</th>
                        <th>Venue</th>
                    </tr>
                    {''.join(f"<tr><td>{row['title']}</td><td>{row['authors']}</td><td>{row['year']}</td><td>{row['citations']}</td><td>{row['venue']}</td></tr>"
                            for _, row in self.df.iterrows())}
                </table>
            </div>
            
            <div class="section">
                <h2>5. Article Abstracts</h2>
                {''.join(f"<h3>{row['title']}</h3><div class='abstract'>{row['abstract']}</div>"
                        for _, row in self.df.iterrows())}
            </div>
        </body>
        </html>
        """
        
        try:
            # Convertir en PDF avec pdfkit
            pdf = pdfkit.from_string(html_content, False)
            
            return dict(
                content=pdf,
                filename=filename,
                type='application/pdf'
            )
        except Exception as e:
            print(f"Erreur lors de la génération du PDF: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Prétraitement du texte"""
        # Tokenisation
        tokens = word_tokenize(text)
        
        # Suppression des stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.lower() not in stop_words]
        
        # Lemmatisation
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Suppression des ponctuations
        tokens = [t for t in tokens if t.isalpha()]
        
        return ' '.join(tokens)
        
    def extract_named_entities(self, text):
        """Extraction des entités nommées"""
        # Tokenisation
        tokens = word_tokenize(text)
        
        # Étiquetage des parties du discours
        tagged = pos_tag(tokens)
        
        # Extraction des entités nommées
        entities = ne_chunk(tagged)
        
        # Récupération des entités nommées
        named_entities = []
        for entity in entities:
            if hasattr(entity, 'label'):
                named_entities.append(entity.label())
        
        return named_entities
        
    def extract_keywords(self, text):
        """Extraction des mots-clés"""
        # Tokenisation
        tokens = word_tokenize(text)
        
        # Suppression des stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t.lower() not in stop_words]
        
        # Lemmatisation
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Suppression des ponctuations
        tokens = [t for t in tokens if t.isalpha()]
        
        # Compter les occurrences des mots
        keyword_counts = Counter(tokens)
        
        # Récupérer les mots-clés les plus fréquents
        keywords = [keyword for keyword, _ in keyword_counts.most_common(10)]
        
        return keywords
        
    def calculate_h_index(self):
        """Calcule l'indice h"""
        citations = self.df['citations'].sort_values(ascending=False)
        h_index = 0
        for i, citation in enumerate(citations):
            if citation >= i + 1:
                h_index = i + 1
            else:
                break
        return h_index
        
    def calculate_i10_index(self):
        """Calcule l'indice i10"""
        citations = self.df['citations'].sort_values(ascending=False)
        i10_index = 0
        for i, citation in enumerate(citations):
            if citation >= 10:
                i10_index = i + 1
            else:
                break
        return i10_index
