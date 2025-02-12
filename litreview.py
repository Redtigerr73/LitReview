#!/usr/bin/env python3
"""
LitReview - Outil d'analyse d'articles scientifiques
"""

import os
import sys
import logging
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.scraper.scholar_scraper import ScholarScraper
from src.taxonomy.sre_taxonomy import SRETaxonomy
from src.taxonomy.taxonomy_integrator import TaxonomyIntegrator
from src.analysis.analyzer import ScientificAnalyzer
from src.utils.config_manager import ConfigManager
from src.utils.error_handler import setup_logger
from src.processor.export_manager import ExportManager
from src.utils.ui import (
    clear_screen, print_header, print_menu, print_success, 
    print_error, print_info, print_warning, get_user_choice,
    get_user_input, get_yes_no, wait_with_spinner, print_progress
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = setup_logger(__name__)

def get_config_path() -> str:
    """Retourne le chemin absolu du fichier de configuration"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config", "settings.yaml")
    
    # Créer le dossier config s'il n'existe pas
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    return config_path

def open_file(filepath):
    """Ouvre un fichier avec l'application par défaut sous Windows"""
    try:
        # Convertir en chemin absolu
        abs_path = os.path.abspath(filepath)
        print_info(f"Ouverture du fichier: {abs_path}")
        
        # Utiliser os.startfile pour Windows
        os.startfile(abs_path)
        
    except Exception as e:
        print_error(f"Impossible d'ouvrir le fichier: {e}")
        print_info("Vous pouvez ouvrir le fichier manuellement à l'emplacement suivant:")
        print_info(abs_path)

def generate_html_report(results, query, timestamp, taxonomy_results=None):
    """Generate a comprehensive HTML report for the systematic literature review."""
    if not results:
        return None

    try:
        # Create the scientific analyzer
        analyzer = ScientificAnalyzer(results)
        analysis = analyzer.analyze_slr()

        # Generate articles table HTML
        articles_table = generate_articles_table_html(results)

        # Format data for JavaScript
        yearly_papers_data = str(analysis['temporal_analysis']['yearly_papers']).replace("'", '"')
        yearly_citations_data = str(analysis['temporal_analysis']['yearly_citations']).replace("'", '"')

        # Generate report filename
        report_filename = f"report_{timestamp}.html"
        report_path = os.path.join("data", report_filename)
        os.makedirs("data", exist_ok=True)

        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'Analyse de Littérature Systématique</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                .container {{ max-width: 1200px; margin: auto; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .section h2 {{ color: #2c3e50; margin-bottom: 20px; }}
                .metric-card {{ background: #f8f9fa; padding: 15px; margin: 10px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                table {{ width: 100%; margin-bottom: 1rem; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .chart {{ margin: 20px 0; height: 400px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center mb-4">Rapport d'Analyse de Littérature Systématique</h1>
                
                <!-- Informations de base -->
                <div class="section">
                    <h2>Informations de base</h2>
                    <p><strong>Requête :</strong> {query}</p>
                    <p><strong>Date de génération :</strong> {timestamp}</p>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value">{analysis['basic_stats']['total_papers']}</div>
                                <div class="metric-label">Articles au total</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value">{analysis['basic_stats']['year_range']}</div>
                                <div class="metric-label">Période couverte</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value">{analysis['basic_stats']['total_citations']}</div>
                                <div class="metric-label">Citations totales</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="metric-card">
                                <div class="metric-value">{analysis['basic_stats']['avg_citations']}</div>
                                <div class="metric-label">Citations moyennes</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Analyse temporelle -->
                <div class="section">
                    <h2>Analyse temporelle</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Distribution annuelle des publications</h4>
                            <div id="yearly-papers-chart" class="chart"></div>
                        </div>
                        <div class="col-md-6">
                            <h4>Évolution des citations</h4>
                            <div id="yearly-citations-chart" class="chart"></div>
                        </div>
                    </div>
                    <div class="mt-4">
                        <h4>Taux de croissance</h4>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>Année</th>
                                    <th>Taux de croissance (%)</th>
                                </tr>
                            </thead>
                            <tbody>
                                {''.join(f"<tr><td>{rate['year']}</td><td>{rate['growth_rate']}%</td></tr>" 
                                        for rate in analysis['temporal_analysis']['growth_rates'][-5:])}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <!-- Analyse des citations -->
                <div class="section">
                    <h2>Analyse des citations</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Distribution des citations</h4>
                            {generate_citation_ranges_html(analysis['citation_analysis']['citation_ranges'])}
                        </div>
                        <div class="col-md-6">
                            <h4>Articles les plus cités</h4>
                            {generate_top_cited_papers_html(analysis['citation_analysis']['top_cited_papers'])}
                        </div>
                    </div>
                </div>
                
                <!-- Analyse des auteurs -->
                <div class="section">
                    <h2>Analyse des auteurs</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Statistiques de collaboration</h4>
                            {generate_author_stats_html(analysis['author_analysis'])}
                        </div>
                        <div class="col-md-6">
                            <h4>Auteurs les plus prolifiques</h4>
                            {generate_top_authors_html(analysis['author_analysis']['top_authors'])}
                        </div>
                    </div>
                </div>
                
                <!-- Analyse des sujets -->
                <div class="section">
                    <h2>Analyse des sujets</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Clusters thématiques</h4>
                            {generate_topic_clusters_html(analysis['topic_analysis']['clusters'])}
                        </div>
                        <div class="col-md-6">
                            <h4>Évolution des sujets</h4>
                            {generate_topic_evolution_html(analysis['topic_analysis']['topic_evolution'])}
                        </div>
                    </div>
                </div>
                
                <!-- Analyse méthodologique -->
                <div class="section">
                    <h2>Analyse méthodologique</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Distribution des méthodologies</h4>
                            {generate_methodology_distribution_html(analysis['methodology_analysis']['methodology_distribution'])}
                        </div>
                        <div class="col-md-6">
                            <h4>Types d'études</h4>
                            <table class="table">
                                <tbody>
                                    <tr>
                                        <td>Études primaires</td>
                                        <td>{analysis['methodology_analysis']['primary_studies']}</td>
                                    </tr>
                                    <tr>
                                        <td>Études secondaires</td>
                                        <td>{analysis['methodology_analysis']['secondary_studies']}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Évaluation de la qualité -->
                <div class="section">
                    <h2>Évaluation de la qualité</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Métriques de qualité</h4>
                            {generate_quality_metrics_html(analysis['quality_assessment']['quality_metrics'])}
                        </div>
                        <div class="col-md-6">
                            <h4>Distribution de la qualité</h4>
                            {generate_quality_distribution_html(analysis['quality_assessment']['quality_distribution'])}
                        </div>
                    </div>
                </div>
                
                <!-- Lacunes de recherche -->
                <div class="section">
                    <h2>Lacunes et directions futures</h2>
                    <div class="row">
                        <div class="col-md-6">
                            <h4>Sujets émergents</h4>
                            {generate_emerging_topics_html(analysis['research_gaps']['emerging_topics'])}
                        </div>
                        <div class="col-md-6">
                            <h4>Zones peu explorées</h4>
                            {generate_underexplored_areas_html(analysis['research_gaps']['underexplored_areas'])}
                        </div>
                    </div>
                </div>
                
                <!-- Liste des articles -->
                <div class="section">
                    <h2>Liste des articles</h2>
                    {articles_table}
                </div>
            </div>
            
            <script>
                // Format data for charts
                const yearlyPapersData = {yearly_papers_data};
                const yearlyCitationsData = {yearly_citations_data};
                
                // Create yearly papers chart
                const yearlyPapersTrace = {{
                    x: Object.keys(yearlyPapersData),
                    y: Object.values(yearlyPapersData),
                    type: 'bar',
                    name: 'Publications'
                }};
                
                Plotly.newPlot('yearly-papers-chart', [yearlyPapersTrace], {{
                    title: 'Distribution annuelle des publications',
                    xaxis: {{ title: 'Année' }},
                    yaxis: {{ title: 'Nombre de publications' }}
                }});
                
                // Create yearly citations chart
                const yearlyCitationsTrace = {{
                    x: Object.keys(yearlyCitationsData),
                    y: Object.values(yearlyCitationsData),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Citations'
                }};
                
                Plotly.newPlot('yearly-citations-chart', [yearlyCitationsTrace], {{
                    title: 'Évolution des citations par année',
                    xaxis: {{ title: 'Année' }},
                    yaxis: {{ title: 'Nombre de citations' }}
                }});
            </script>
        </body>
        </html>
        """

        # Write HTML content to file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    except Exception as e:
        logging.error(f"Error generating HTML report: {str(e)}")
        return None

def generate_citation_ranges_html(ranges):
    """Génère le HTML pour les plages de citations"""
    if not ranges:
        return ""
    
    html = "<table class='table'><thead><tr><th>Plage</th><th>Nombre d'articles</th></tr></thead><tbody>"
    for range_label, count in ranges.items():
        html += f"<tr><td>{range_label}</td><td>{count}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_top_cited_papers_html(papers, limit=5):
    """Génère le HTML pour les articles les plus cités"""
    if not papers:
        return ""
    
    html = "<table class='table'><thead><tr><th>Titre</th><th>Citations</th></tr></thead><tbody>"
    for paper in papers[:limit]:
        html += f"<tr><td>{paper['title']}</td><td>{paper['citations']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_author_stats_html(stats):
    """Génère le HTML pour les statistiques d'auteurs"""
    if not stats:
        return ""
    
    html = "<table class='table'><tbody>"
    html += f"<tr><td>Nombre total d'auteurs</td><td>{stats.get('total_authors', 0)}</td></tr>"
    html += f"<tr><td>Moyenne d'auteurs par article</td><td>{stats.get('avg_authors_per_paper', 0)}</td></tr>"
    
    # Accéder aux statistiques de collaboration de manière sécurisée
    collaboration_stats = stats.get('collaboration_stats', {})
    html += f"<tr><td>Articles à auteur unique</td><td>{collaboration_stats.get('single_author_papers', 0)}</td></tr>"
    html += f"<tr><td>Articles en collaboration</td><td>{collaboration_stats.get('multi_author_papers', 0)}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_top_authors_html(authors, limit=5):
    """Génère le HTML pour les auteurs les plus prolifiques"""
    if not authors:
        return ""
    
    html = "<table class='table'><thead><tr><th>Auteur</th><th>Articles</th><th>Citations</th></tr></thead><tbody>"
    for author in authors[:limit]:
        html += f"<tr><td>{author['name']}</td><td>{author['papers']}</td><td>{author['total_citations']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_topic_clusters_html(clusters):
    """Génère le HTML pour les clusters thématiques"""
    if not clusters:
        return ""
    
    html = "<table class='table'><thead><tr><th>Cluster</th><th>Termes principaux</th><th>Taille</th></tr></thead><tbody>"
    for cluster in clusters:
        html += f"<tr><td>{cluster['cluster']}</td><td>{', '.join(cluster['terms'])}</td><td>{cluster['size']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_topic_evolution_html(evolution):
    """Génère le HTML pour l'évolution des sujets"""
    if not evolution:
        return ""
    
    html = "<table class='table'><thead><tr><th>Année</th><th>Sujets dominants</th></tr></thead><tbody>"
    for year_data in evolution:
        topics = [topic['topic'] for topic in year_data['topics'][:3]]
        html += f"<tr><td>{year_data['year']}</td><td>{', '.join(topics)}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_methodology_distribution_html(distribution):
    """Génère le HTML pour la distribution des méthodologies"""
    if not distribution:
        return ""
    
    html = "<table class='table'><thead><tr><th>Méthodologie</th><th>Nombre d'articles</th></tr></thead><tbody>"
    for method, count in distribution.items():
        html += f"<tr><td>{method}</td><td>{count}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_quality_metrics_html(metrics):
    """Génère le HTML pour les métriques de qualité"""
    if not metrics:
        return ""
    
    html = "<table class='table'><tbody>"
    html += f"<tr><td>Articles avec résumé</td><td>{metrics['has_abstract']}</td></tr>"
    html += f"<tr><td>Articles cités</td><td>{metrics['has_citations']}</td></tr>"
    html += f"<tr><td>Articles récents (≥ 2020)</td><td>{metrics['recent_papers']}</td></tr>"
    html += f"<tr><td>Articles à fort impact</td><td>{metrics['high_impact']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_quality_distribution_html(distribution):
    """Génère le HTML pour la distribution de la qualité"""
    if not distribution:
        return ""
    
    html = "<table class='table'><tbody>"
    html += f"<tr><td>Qualité élevée</td><td>{distribution['high']}</td></tr>"
    html += f"<tr><td>Qualité moyenne</td><td>{distribution['medium']}</td></tr>"
    html += f"<tr><td>Qualité faible</td><td>{distribution['low']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_emerging_topics_html(topics, limit=5):
    """Génère le HTML pour les sujets émergents"""
    if not topics:
        return ""
    
    html = "<table class='table'><thead><tr><th>Sujet</th><th>Score d'émergence</th></tr></thead><tbody>"
    for topic in topics[:limit]:
        html += f"<tr><td>{topic['topic']}</td><td>{topic['score']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_underexplored_areas_html(areas):
    """Génère le HTML pour les zones peu explorées"""
    if not areas:
        return ""
    
    html = "<table class='table'><thead><tr><th>Domaine</th><th>Articles</th><th>Citations moyennes</th></tr></thead><tbody>"
    for area in areas:
        html += f"<tr><td>{area['topic']}</td><td>{area['papers']}</td><td>{area['avg_citations']}</td></tr>"
    html += "</tbody></table>"
    return html

def generate_articles_table_html(articles):
    """Génère le HTML pour la table des articles"""
    if not articles:
        return ""
    
    html = "<table class='table'><thead><tr><th>Titre</th><th>Auteurs</th><th>Année</th><th>Citations</th><th>Venue</th></tr></thead><tbody>"
    for article in articles:
        html += f"<tr><td>{article['title']}</td><td>{article['authors']}</td><td>{article['year']}</td><td>{article['citations']}</td><td>{article.get('venue', 'N/A')}</td></tr>"
    html += "</tbody></table>"
    return html

def search_articles(config_manager: ConfigManager):
    """Effectue une recherche d'articles"""
    try:
        # Demander les paramètres de recherche
        query = get_user_input("Recherche: ")
        if not query.strip():
            print_warning("\nVeuillez entrer une recherche valide.")
            return
            
        num_results = get_user_input("Nombre d'articles (défaut: 100)", default="100")
        try:
            num_results = int(num_results)
            if num_results <= 0:
                raise ValueError("Le nombre d'articles doit être positif")
        except ValueError:
            print_warning("\nNombre d'articles invalide. Utilisation de la valeur par défaut (100).")
            num_results = 100
            
        # Initialiser le scraper
        print_info("\nInitialisation du scraper...")
        scraper = ScholarScraper(config_manager)
        
        # Effectuer la recherche
        print_info("Recherche en cours...")
        with wait_with_spinner("Recherche en cours...") as spinner:
            results = scraper.search(query, num_results)
            
        if not results:
            print_warning("\nAucun résultat trouvé.")
            return
            
        # Initialiser l'analyse taxonomique
        print_info("\nAnalyse taxonomique en cours...")
        taxonomy = SRETaxonomy()
        integrator = TaxonomyIntegrator(taxonomy)
        
        with wait_with_spinner("Analyse taxonomique en cours...") as spinner:
            # Convert results to DataFrame for taxonomy processing
            df = pd.DataFrame(results)
            if len(df) > 0:
                taxonomy_results = integrator.process_papers(df)
            else:
                taxonomy_results = {}
            
        # Générer le timestamp pour les fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les résultats bruts
        filename = f"results_{timestamp}.xlsx"
        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        
        df = pd.DataFrame(results)
        if len(df) > 0:
            df.to_excel(filepath, index=False)
            print_success(f"\n{len(results)} articles sauvegardés dans {filepath}")
            
            # Générer et ouvrir le rapport
            print_info("\nGénération du rapport...")
            try:
                report_path = generate_html_report(results, query, timestamp, taxonomy_results)
                
                if report_path:
                    print_success(f"Rapport généré : {report_path}")
                    if get_yes_no("Voulez-vous ouvrir le rapport maintenant?"):
                        open_file(report_path)
            except Exception as e:
                print_error(f"Impossible de générer le rapport: {str(e)}")
                logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
        else:
            print_warning("\nAucun résultat à sauvegarder.")
                
    except KeyboardInterrupt:
        print_info("\n\nRecherche annulée.")
    except Exception as e:
        print_error(f"\nErreur lors de la recherche : {str(e)}")
        logger.error(f"Erreur détaillée : {traceback.format_exc()}")
    finally:
        if 'scraper' in locals():
            scraper.cleanup()

def analyze_results():
    """Analyse les résultats existants"""
    print_info("\nFonctionnalité d'analyse en développement...")

def main():
    """Point d'entrée principal"""
    try:
        # Charger la configuration
        config_path = get_config_path()
        logger.info(f"Using config from: {config_path}")
        config_manager = ConfigManager(config_path)
        
        while True:
            clear_screen()
            print_header("LitReview - Outil d'analyse d'articles scientifiques")
            print_menu([
                "Rechercher des articles",
                "Analyser des résultats",
                "Quitter"
            ])
            
            choice = get_user_choice(1, 3)
            
            if choice == 3:
                print_info("\nAu revoir !")
                break
                
            elif choice == 1:
                search_articles(config_manager)
                input("\nAppuyez sur Entrée pour continuer...")
                
            elif choice == 2:
                analyze_results()
                input("\nAppuyez sur Entrée pour continuer...")
                
    except KeyboardInterrupt:
        print_info("\n\nAu revoir !")
    except Exception as e:
        print_error(f"\nErreur fatale : {str(e)}")
        logger.error(f"Erreur détaillée : {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
