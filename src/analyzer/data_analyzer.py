"""
Data analysis and visualization module for LitReview
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # ou le nombre de cores que vous souhaitez utiliser
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from datetime import datetime
import logging
import traceback
import io
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

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

class LitReviewAnalyzer:
    """
    Analyzer for scholarly article data
    """
    
    def __init__(self):
        """Initialize the analyzer"""
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
            
            # Vérifier si le fichier existe
            if not os.path.exists(file_path):
                logger.error(f"Le fichier {file_path} n'existe pas")
                return False
                
            # Charger les données selon le format
            if file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                logger.error("Format de fichier non supporté. Utilisez .xlsx ou .csv")
                return False
                
            # Vérifier si les données sont vides
            if self.data is None or len(self.data) == 0:
                logger.error("Le fichier de données est vide")
                return False
                
            # Clean and prepare data
            self._prepare_data()
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}\n{traceback.format_exc()}")
            return False
            
    def _prepare_data(self) -> None:
        """
        Clean and prepare data for analysis
        """
        try:
            if self.data is None or len(self.data) == 0:
                return
                
            # Convert year to numeric, handling errors
            if 'year' in self.data.columns:
                logger.debug("Converting year to numeric")
                self.data['year'] = pd.to_numeric(self.data['year'], errors='coerce')
                
            # Convert citations to numeric, handling errors
            if 'citations' in self.data.columns:
                logger.debug("Converting citations to numeric")
                self.data['citations'] = pd.to_numeric(self.data['citations'], errors='coerce').fillna(0)
                
            # Convert retrieved_date to datetime
            if 'retrieved_date' in self.data.columns:
                logger.debug("Converting retrieved_date to datetime")
                self.data['retrieved_date'] = pd.to_datetime(self.data['retrieved_date'], errors='coerce')
                
            # Clean text fields
            text_columns = ['title', 'abstract', 'publisher']
            for col in text_columns:
                if col in self.data.columns:
                    logger.debug(f"Cleaning text field: {col}")
                    self.data[col] = self.data[col].fillna('').astype(str)
                    
        except Exception as e:
            logger.error(f"Erreur lors de la préparation des données: {str(e)}\n{traceback.format_exc()}")
            
    def _safe_get_value(self, series: pd.Series, operation: str) -> Any:
        """Safely get a value from a pandas Series"""
        try:
            if len(series) == 0:
                return 0 if operation in ['sum', 'mean'] else 'N/A'
                
            if operation == 'sum':
                return int(series.sum())
            elif operation == 'mean':
                return round(float(series.mean()), 2)
            elif operation == 'min':
                return int(series.min())
            elif operation == 'max':
                return int(series.max())
            else:
                return 'N/A'
        except:
            return 'N/A'
            
    def generate_summary(self) -> Dict:
        """
        Generate a summary of the dataset
        """
        try:
            logger.info("Starting generate_summary...")
            logger.debug(f"DataFrame shape: {self.data.shape if self.data is not None else 'None'}")
            logger.debug(f"DataFrame columns: {self.data.columns.tolist() if self.data is not None else 'None'}")
            logger.debug(f"DataFrame info:")
            if self.data is not None:
                buffer = io.StringIO()
                self.data.info(buf=buffer)
                logger.debug(buffer.getvalue())
            
            if self.data is None:
                logger.error("Data is None")
                return {}
                
            if len(self.data) == 0:
                logger.error("Data has 0 rows")
                return {}
                
            summary = {'total_articles': len(self.data)}
            logger.debug(f"Total articles: {summary['total_articles']}")
            
            # Year range
            if 'year' in self.data.columns:
                logger.debug("Processing year data...")
                logger.debug(f"Year values before filtering: {self.data['year'].value_counts().to_dict()}")
                
                # Only consider years > 0
                year_data = self.data[self.data['year'] > 0]['year']
                year_count = len(year_data)
                logger.debug(f"Valid year data count: {year_count}")
                logger.debug(f"Year values after filtering: {year_data.value_counts().to_dict()}")
                
                if year_count > 0:
                    min_year = int(year_data.min())
                    max_year = int(year_data.max())
                    summary['year_range'] = f"{min_year} - {max_year}"
                    logger.debug(f"Year range: {summary['year_range']}")
                else:
                    summary['year_range'] = "Années non disponibles"
                    logger.debug("No valid year data found")
            else:
                summary['year_range'] = "Années non disponibles"
                logger.debug("No year column found")
                
            # Citations
            if 'citations' in self.data.columns:
                logger.debug("Processing citation data...")
                logger.debug(f"Citation values before filling NA: {self.data['citations'].value_counts().to_dict()}")
                citations_data = self.data['citations'].fillna(0)
                logger.debug(f"Citation values after filling NA: {citations_data.value_counts().to_dict()}")
                
                summary['total_citations'] = int(citations_data.sum())
                summary['avg_citations'] = round(float(citations_data.mean()), 2)
                logger.debug(f"Citations - Total: {summary['total_citations']}, Avg: {summary['avg_citations']}")
                
            # Publishers
            if 'publisher' in self.data.columns:
                logger.debug("Processing publisher data...")
                logger.debug(f"Publisher values before dropna: {self.data['publisher'].value_counts().to_dict()}")
                publisher_data = self.data['publisher'].dropna()
                publisher_count = len(publisher_data)
                logger.debug(f"Publisher data count: {publisher_count}")
                logger.debug(f"Publisher values after dropna: {publisher_data.value_counts().to_dict()}")
                
                if publisher_count > 0:
                    publisher_counts = publisher_data.value_counts()
                    summary['top_publishers'] = {str(k): int(v) for k, v in 
                                              publisher_counts.head(5).items()}
                    logger.debug(f"Top publishers: {summary['top_publishers']}")
                else:
                    summary['top_publishers'] = {}
                    logger.debug("No valid publisher data found")
                    
            logger.info("Summary generation completed successfully")
            logger.debug(f"Final summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in generate_summary: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
            
    def plot_year_distribution(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot distribution of articles by year
        """
        try:
            if self.data is None or len(self.data) == 0:
                logger.error("Aucune donnée disponible pour la distribution par année")
                return None
                
            if 'year' not in self.data.columns:
                logger.error("Colonne 'year' non trouvée")
                return None
                
            # Filter out years that are 0 (unknown)
            valid_years = self.data[self.data['year'] > 0]['year']
            
            if len(valid_years) == 0:
                logger.warning("Aucune année valide trouvée dans les données")
                return None
                
            plt.figure(figsize=(12, 6))
            sns.histplot(data=valid_years, bins=20)
            plt.title('Distribution des Articles par Année')
            plt.xlabel('Année')
            plt.ylabel('Nombre d\'Articles')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                return save_path
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique de distribution par année: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    def plot_citation_distribution(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot distribution of citations
        """
        try:
            if self.data is None or len(self.data) == 0:
                logger.error("Aucune donnée disponible pour la distribution des citations")
                return None
                
            if 'citations' not in self.data.columns:
                logger.error("Pas de données de citations disponibles")
                return None
                
            plt.figure(figsize=(12, 6))
            sns.histplot(data=self.data, x='citations', bins=30)
            plt.title('Distribution des Citations')
            plt.xlabel('Nombre de Citations')
            plt.ylabel('Nombre d\'Articles')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                return save_path
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du graphique de distribution des citations: {str(e)}\n{traceback.format_exc()}")
            return None
            
    def generate_wordcloud(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Generate word cloud from titles and abstracts
        """
        try:
            logger.info("Starting wordcloud generation...")
            logger.debug(f"Data shape: {self.data.shape if self.data is not None else 'None'}")
            
            if self.data is None or len(self.data) == 0:
                logger.error("No data available for wordcloud")
                return None
                
            # Combine titles and abstracts
            text = ""
            if 'title' in self.data.columns:
                logger.debug("Processing titles for wordcloud")
                titles = self.data['title'].fillna('')
                text += ' '.join(titles)
                
            if 'abstract' in self.data.columns:
                logger.debug("Processing abstracts for wordcloud")
                abstracts = self.data['abstract'].fillna('')
                text += ' ' + ' '.join(abstracts)
                
            if not text.strip():
                logger.error("No text available for wordcloud")
                return None
                
            logger.debug(f"Total text length for wordcloud: {len(text)}")
            
            # Generate word cloud
            wordcloud = WordCloud(width=1200, height=800, 
                                background_color='white',
                                max_words=100).generate(text)
                                
            # Create figure
            plt.figure(figsize=(12, 8))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            if save_path:
                logger.debug(f"Saving wordcloud to {save_path}")
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                return save_path
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating wordcloud: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def export_analysis(self, output_dir: str) -> Optional[str]:
        """
        Export complete analysis to HTML report
        """
        try:
            logger.info("Starting export_analysis...")
            logger.debug(f"DataFrame state at start of export_analysis:")
            logger.debug(f"- Shape: {self.data.shape if self.data is not None else 'None'}")
            logger.debug(f"- Columns: {self.data.columns.tolist() if self.data is not None else 'None'}")
            logger.debug(f"- Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB" if self.data is not None else "N/A")
            
            if self.data is None or len(self.data) == 0:
                logger.error("No data available for analysis")
                return None
                
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created/verified output directory: {output_path}")
            
            # Create plots directory
            plots_dir = output_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            logger.debug(f"Created/verified plots directory: {plots_dir}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            logger.debug(f"Using timestamp: {timestamp}")
            
            # Generate year distribution plot
            logger.debug("Starting year distribution plot generation...")
            year_plot_path = str(plots_dir / 'year_distribution.png')
            year_plot = None
            
            if 'year' in self.data.columns:
                logger.debug("Found year column")
                valid_years = self.data[self.data['year'] > 0]['year']
                logger.debug(f"Valid years count: {len(valid_years)}")
                logger.debug(f"Year values: {valid_years.value_counts().to_dict()}")
                
                if len(valid_years) > 0:
                    year_plot = self.plot_year_distribution(year_plot_path)
                    logger.debug(f"Year plot generated: {year_plot_path if year_plot else 'Failed'}")
            else:
                logger.debug("No year column found")
                
            # Generate citation distribution plot
            logger.debug("Starting citation distribution plot generation...")
            citation_plot_path = str(plots_dir / 'citation_distribution.png')
            citation_plot = None
            
            if 'citations' in self.data.columns:
                logger.debug("Found citations column")
                citations_data = self.data['citations'].fillna(0)
                logger.debug(f"Citations data - Mean: {citations_data.mean():.2f}, Max: {citations_data.max()}")
                
                if not citations_data.empty:
                    citation_plot = self.plot_citation_distribution(citation_plot_path)
                    logger.debug(f"Citation plot generated: {citation_plot_path if citation_plot else 'Failed'}")
            else:
                logger.debug("No citations column found")
                
            # Generate wordcloud
            logger.debug("Starting wordcloud generation...")
            wordcloud_path = str(plots_dir / 'wordcloud.png')
            wordcloud = None
            
            text_available = False
            text_stats = {}
            
            if 'title' in self.data.columns:
                title_data = self.data['title'].fillna('')
                text_stats['title'] = {
                    'empty_count': (title_data == '').sum(),
                    'total_chars': title_data.str.len().sum()
                }
                if not title_data.empty and text_stats['title']['total_chars'] > 0:
                    text_available = True
                    
            if 'abstract' in self.data.columns:
                abstract_data = self.data['abstract'].fillna('')
                text_stats['abstract'] = {
                    'empty_count': (abstract_data == '').sum(),
                    'total_chars': abstract_data.str.len().sum()
                }
                if not abstract_data.empty and text_stats['abstract']['total_chars'] > 0:
                    text_available = True
                    
            logger.debug(f"Text statistics: {text_stats}")
            
            if text_available:
                wordcloud = self.generate_wordcloud(wordcloud_path)
                logger.debug(f"Wordcloud generated: {wordcloud_path if wordcloud else 'Failed'}")
            else:
                logger.debug("No text available for wordcloud")
                
            # Generate summary
            logger.debug("Starting summary generation...")
            summary = self.generate_summary()
            if not summary:
                logger.error("Failed to generate summary")
                return None
            logger.debug(f"Generated summary: {summary}")
            
            # Create HTML report
            logger.info("Creating HTML report...")
            report_path = output_path / f"{Path(self.file_path).stem}_analysis_{timestamp}.html"
            logger.debug(f"Report path: {report_path}")
            
            logger.debug("Preparing HTML report parameters:")
            logger.debug(f"- Year plot path: {year_plot_path if year_plot else 'None'}")
            logger.debug(f"- Citation plot path: {citation_plot_path if citation_plot else 'None'}")
            logger.debug(f"- Wordcloud path: {wordcloud_path if wordcloud else 'None'}")
            
            html_content = self._generate_html_report(
                summary=summary,
                year_plot=year_plot_path if year_plot else None,
                citation_plot=citation_plot_path if citation_plot else None,
                wordcloud=wordcloud_path if wordcloud else None
            )
            
            if not html_content:
                logger.error("HTML content generation failed")
                return None
                
            logger.debug(f"Writing HTML content ({len(html_content)} chars) to {report_path}")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Report generated successfully at {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error in export_analysis: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def _generate_html_report(self, summary: Dict, year_plot: Optional[str], 
                            citation_plot: Optional[str], wordcloud: Optional[str]) -> str:
        """
        Generate HTML report content
        """
        try:
            logger.info("Starting HTML report generation...")
            
            # Log input validation
            logger.debug("Validating inputs:")
            logger.debug(f"summary type: {type(summary)}")
            logger.debug(f"summary content: {summary}")
            logger.debug(f"year_plot: {year_plot}")
            logger.debug(f"citation_plot: {citation_plot}")
            logger.debug(f"wordcloud: {wordcloud}")
            
            if not isinstance(summary, dict):
                logger.error(f"Invalid summary type: {type(summary)}")
                return ""
                
            # Convert paths to relative paths for HTML
            def get_relative_path(path: Optional[str]) -> Optional[str]:
                if not path:
                    logger.debug(f"No path provided for conversion")
                    return None
                rel_path = os.path.join('plots', os.path.basename(path))
                logger.debug(f"Converted {path} to {rel_path}")
                return rel_path
                
            year_plot_rel = get_relative_path(year_plot)
            citation_plot_rel = get_relative_path(citation_plot)
            wordcloud_rel = get_relative_path(wordcloud)
            
            logger.debug("Relative paths:")
            logger.debug(f"- year_plot: {year_plot_rel}")
            logger.debug(f"- citation_plot: {citation_plot_rel}")
            logger.debug(f"- wordcloud: {wordcloud_rel}")
            
            # Start building HTML content
            logger.debug("Building HTML content...")
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="fr">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Rapport d'Analyse LitReview</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        margin: 40px;
                        line-height: 1.6;
                        color: #333;
                    }}
                    .container {{ 
                        max-width: 1200px; 
                        margin: 0 auto; 
                        padding: 20px;
                    }}
                    .section {{ 
                        margin-bottom: 40px;
                        background: #fff;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    h1, h2 {{ 
                        color: #2c3e50;
                        margin-bottom: 20px;
                    }}
                    img {{ 
                        max-width: 100%; 
                        height: auto;
                        border-radius: 4px;
                        margin: 10px 0;
                    }}
                    table {{ 
                        width: 100%;
                        border-collapse: collapse;
                        margin: 10px 0;
                    }}
                    th, td {{ 
                        padding: 12px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }}
                    th {{
                        background-color: #f8f9fa;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Rapport d'Analyse LitReview</h1>
                    
                    <div class="section">
                        <h2>Résumé</h2>
                        <table>
                            <tr><th>Métrique</th><th>Valeur</th></tr>
            """
            
            # Add summary metrics with logging
            logger.debug("Adding summary metrics to HTML")
            metrics = {
                'Nombre total d\'articles': summary.get('total_articles', 'N/A'),
                'Période couverte': summary.get('year_range', 'N/A'),
                'Total des citations': summary.get('total_citations', 'N/A'),
                'Moyenne des citations': summary.get('avg_citations', 'N/A')
            }
            
            logger.debug(f"Metrics to add: {metrics}")
            for metric, value in metrics.items():
                html_content += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                logger.debug(f"Added metric: {metric} = {value}")
                
            html_content += """
                        </table>
                    </div>
            """
            
            # Add plots with logging
            if year_plot_rel:
                logger.debug(f"Adding year plot with path: {year_plot_rel}")
                html_content += f"""
                    <div class="section">
                        <h2>Distribution par Année</h2>
                        <img src="{year_plot_rel}" alt="Distribution par Année">
                    </div>
                """
                
            if citation_plot_rel:
                logger.debug(f"Adding citation plot with path: {citation_plot_rel}")
                html_content += f"""
                    <div class="section">
                        <h2>Distribution des Citations</h2>
                        <img src="{citation_plot_rel}" alt="Distribution des Citations">
                    </div>
                """
                
            if wordcloud_rel:
                logger.debug(f"Adding wordcloud with path: {wordcloud_rel}")
                html_content += f"""
                    <div class="section">
                        <h2>Nuage de Mots des Titres et Résumés</h2>
                        <img src="{wordcloud_rel}" alt="Nuage de Mots">
                    </div>
                """
                
            # Add publisher information with logging
            logger.debug("Adding publisher information")
            if 'top_publishers' in summary:
                logger.debug(f"Found top_publishers in summary: {summary['top_publishers']}")
                if summary['top_publishers']:
                    html_content += """
                        <div class="section">
                            <h2>Top 5 des Éditeurs</h2>
                            <table>
                                <tr><th>Éditeur</th><th>Nombre d'Articles</th></tr>
                    """
                    
                    for publisher, count in summary['top_publishers'].items():
                        if publisher and count:  # Vérifie que les valeurs ne sont pas None
                            html_content += f"<tr><td>{publisher}</td><td>{count}</td></tr>"
                            logger.debug(f"Added publisher: {publisher} = {count}")
                            
                    html_content += """
                            </table>
                        </div>
                    """
                else:
                    logger.debug("top_publishers is empty")
            else:
                logger.debug("No top_publishers in summary")
                
            html_content += """
                </div>
            </body>
            </html>
            """
            
            logger.info("HTML report generation completed successfully")
            logger.debug(f"Generated HTML content length: {len(html_content)}")
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return ""

class ScientificAnalyzer:
    """Analyseur scientifique pour SLR (Systematic Literature Review)"""
    
    def __init__(self, results):
        """Initialise l'analyseur avec les résultats"""
        self.df = pd.DataFrame(results)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        self.prepare_data()
        
    def prepare_data(self):
        """Prépare les données pour l'analyse"""
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
        self.lemmatizer = WordNetLemmatizer()
        self.df['processed_text'] = self.df.apply(
            lambda x: self.preprocess_text(str(x['title']) + ' ' + str(x['abstract'])), 
            axis=1
        )
        
        # Extraire les entités nommées et mots-clés
        self.df['named_entities'] = self.df['processed_text'].apply(self.extract_named_entities)
        self.df['keywords'] = self.df['processed_text'].apply(self.extract_keywords)
        
        # Calculer les métriques
        self.compute_metrics()
        
    def analyze_slr(self):
        """Effectue une analyse complète pour SLR"""
        analysis = {
            'basic_stats': self.get_basic_stats(),
            'temporal_analysis': self.analyze_temporal_trends(),
            'citation_analysis': self.analyze_citations(),
            'author_analysis': self.analyze_authors(),
            'topic_analysis': self.analyze_topics(),
            'methodology_analysis': self.analyze_methodology(),
            'quality_assessment': self.assess_quality(),
            'research_gaps': self.identify_research_gaps()
        }
        return analysis
        
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
        """Analyse les auteurs des articles"""
        try:
            # Calculer le nombre total d'auteurs uniques
            all_authors = []
            authors_papers = {}
            authors_citations = {}
            single_author_papers = 0
            multi_author_papers = 0

            for _, row in self.df.iterrows():
                authors = row['authors'].split(', ') if isinstance(row['authors'], str) else []
                citations = row['citations'] if pd.notnull(row['citations']) else 0
                
                # Compter les articles à auteur unique vs collaborations
                if len(authors) == 1:
                    single_author_papers += 1
                elif len(authors) > 1:
                    multi_author_papers += 1
                
                # Mettre à jour les statistiques par auteur
                for author in authors:
                    if author and author not in all_authors:
                        all_authors.append(author)
                    authors_papers[author] = authors_papers.get(author, 0) + 1
                    authors_citations[author] = authors_citations.get(author, 0) + citations

            # Calculer la moyenne d'auteurs par article
            author_counts = self.df['authors'].apply(lambda x: len(x.split(', ')) if isinstance(x, str) else 0)
            avg_authors = author_counts.mean() if not author_counts.empty else 0

            # Préparer les top auteurs
            top_authors = []
            for author in authors_papers:
                if author:  # Ignorer les auteurs vides
                    top_authors.append({
                        'name': author,
                        'papers': authors_papers[author],
                        'total_citations': authors_citations[author]
                    })
            top_authors.sort(key=lambda x: (x['papers'], x['total_citations']), reverse=True)

            return {
                'total_authors': len(all_authors),
                'avg_authors_per_paper': round(avg_authors, 2),
                'collaboration_stats': {
                    'single_author_papers': single_author_papers,
                    'multi_author_papers': multi_author_papers
                },
                'top_authors': top_authors[:10]  # Limiter aux 10 premiers auteurs
            }
        except Exception as e:
            logging.error(f"Error in analyze_authors: {str(e)}")
            return {
                'total_authors': 0,
                'avg_authors_per_paper': 0,
                'collaboration_stats': {
                    'single_author_papers': 0,
                    'multi_author_papers': 0
                },
                'top_authors': []
            }
        
    def analyze_topics(self):
        """Analyse des sujets et tendances thématiques"""
        try:
            # Préparation des données textuelles
            texts = []
            for _, row in self.df.iterrows():
                title = str(row['title']) if pd.notnull(row['title']) else ''
                abstract = str(row['abstract']) if pd.notnull(row['abstract']) else ''
                texts.append(f"{title} {abstract}")

            # Vectorisation TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Clustering K-means avec n_init explicite
            n_clusters = min(8, len(self.df))
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,  # Spécifier explicitement n_init
                random_state=42
            )
            clusters = kmeans.fit_predict(tfidf_matrix)

            # Extraire les termes principaux pour chaque cluster
            terms = vectorizer.get_feature_names_out()
            cluster_terms = []
            for i in range(n_clusters):
                if len(texts) > 0:  # Vérifier qu'il y a des textes à analyser
                    cluster_docs = tfidf_matrix[clusters == i]
                    if cluster_docs.shape[0] > 0:  # Vérifier qu'il y a des documents dans le cluster
                        centroid = cluster_docs.mean(axis=0).A1
                        top_term_indices = centroid.argsort()[-5:][::-1]
                        cluster_terms.append({
                            'cluster': i,
                            'terms': [terms[idx] for idx in top_term_indices],
                            'size': int((clusters == i).sum())
                        })

            # Analyse de l'évolution temporelle des sujets
            years = sorted(self.df['year'].unique())
            topic_evolution = []
            for year in years:
                year_docs = self.df[self.df['year'] == year]
                if not year_docs.empty:
                    year_topics = self._extract_year_topics(year_docs)
                    topic_evolution.append({
                        'year': int(year),
                        'topics': year_topics
                    })

            return {
                'clusters': cluster_terms,
                'topic_evolution': topic_evolution
            }

        except Exception as e:
            logging.error(f"Error in analyze_topics: {str(e)}")
            return {
                'clusters': [],
                'topic_evolution': []
            }
        
    def analyze_methodology(self):
        """Analyse des méthodologies de recherche"""
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
            text = f"{row['title']} {row['abstract']}".lower()
            for method, keywords in methodology_keywords.items():
                if any(keyword in text for keyword in keywords):
                    methodology_counts[method] += 1
        
        return {
            'methodology_distribution': dict(methodology_counts),
            'primary_studies': len(self.df) - methodology_counts['review'],
            'secondary_studies': methodology_counts['review']
        }
        
    def assess_quality(self):
        """Évaluation de la qualité des études"""
        quality_metrics = {
            'has_abstract': self.df['abstract'].notna().sum(),
            'has_citations': (self.df['citations'] > 0).sum(),
            'recent_papers': (self.df['year'] >= 2020).sum(),
            'high_impact': (self.df['citations'] >= self.df['citations'].quantile(0.75)).sum()
        }
        
        return {
            'quality_metrics': quality_metrics,
            'quality_distribution': {
                'high': int((self.df['citations'] >= self.df['citations'].quantile(0.75)).sum()),
                'medium': int((self.df['citations'] >= self.df['citations'].quantile(0.25)).sum()),
                'low': int((self.df['citations'] < self.df['citations'].quantile(0.25)).sum())
            }
        }
        
    def identify_research_gaps(self):
        """Identification des lacunes de recherche"""
        recent_years = self.df[self.df['year'] >= 2020]
        emerging_topics = self.analyze_emerging_topics(recent_years)
        
        return {
            'emerging_topics': emerging_topics,
            'underexplored_areas': self.find_underexplored_areas(),
            'future_directions': self.suggest_future_directions()
        }
        
    def analyze_emerging_topics(self, recent_df):
        """Analyse des sujets émergents"""
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
        
    def find_underexplored_areas(self):
        """Identifie les domaines peu explorés"""
        # Analyser les zones avec peu de publications mais citations élevées
        topic_impact = defaultdict(lambda: {'papers': 0, 'citations': 0})
        
        for _, row in self.df.iterrows():
            for keyword in row.get('keywords', []):
                topic_impact[keyword]['papers'] += 1
                topic_impact[keyword]['citations'] += row['citations']
        
        underexplored = []
        for topic, stats in topic_impact.items():
            if stats['papers'] < 3 and stats['citations'] > 0:  # Critères arbitraires
                underexplored.append({
                    'topic': topic,
                    'papers': stats['papers'],
                    'avg_citations': round(stats['citations'] / stats['papers'], 2)
                })
        
        return sorted(underexplored, key=lambda x: x['avg_citations'], reverse=True)[:5]
        
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

    def _extract_year_topics(self, year_df):
        """Analyse des sujets pour une année donnée"""
        if len(year_df) == 0:
            return []
            
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(year_df['processed_text'])
        
        feature_names = tfidf.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        
        top_topics = []
        for idx in scores.argsort()[-5:][::-1]:
            top_topics.append({
                'topic': feature_names[idx],
                'score': round(float(scores[idx]), 4)
            })
            
        return top_topics
