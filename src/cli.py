"""
Interactive CLI interface for LitReview
"""

import os
import sys
from typing import Optional, List, Dict
import logging
import traceback
from datetime import datetime
from tqdm import tqdm
from colorama import init, Fore, Style

from .scraper.scholar_scraper import ScholarScraper
from .processor.export_manager import ExportManager
from .utils.config_manager import ConfigManager
from .utils.error_handler import setup_logger
from .analyzer.data_analyzer import LitReviewAnalyzer

# Initialize colorama for Windows color support
init()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = setup_logger(__name__)

def print_header(text: str) -> None:
    """Affiche un texte formaté comme en-tête"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")

def print_success(text: str) -> None:
    """Affiche un message de succès"""
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text: str) -> None:
    """Affiche un message d'erreur"""
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text: str) -> None:
    """Affiche un message d'information"""
    print(f"{Fore.BLUE}ℹ {text}{Style.RESET_ALL}")

def get_config_path() -> str:
    """Retourne le chemin absolu du fichier de configuration"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", "settings.yaml")
    
    # Créer le dossier config s'il n'existe pas
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    return config_path

def search_articles(query: str, num_results: int) -> Optional[List[Dict]]:
    """Effectue une recherche d'articles avec gestion des erreurs"""
    scraper = None
    try:
        # Initialiser le scraper avec le chemin de configuration
        config_path = get_config_path()
        logger.info(f"Using config from: {config_path}")
        
        config_manager = ConfigManager(config_path)
        scraper = ScholarScraper(config_manager)
        
        print_info("Initialisation de la recherche...")
        with tqdm(total=100, desc="Recherche en cours", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            results = scraper.search(query, num_results)
            pbar.update(100)
        
        if not results:
            print_error("Aucun résultat trouvé.")
            return None
            
        print_success(f"{len(results)} articles trouvés")
        return results
        
    except Exception as e:
        print_error(f"Erreur lors de la recherche : {str(e)}")
        logger.error(f"Erreur détaillée : {traceback.format_exc()}")
        return None
        
    finally:
        if scraper:
            try:
                scraper.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

def save_results(results: List[Dict], filename: str) -> bool:
    """Sauvegarde les résultats avec gestion des erreurs"""
    try:
        # Créer le dossier data s'il n'existe pas
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", filename)
        
        print_info("Sauvegarde des résultats en cours...")
        with tqdm(total=1, desc="Export Excel", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            # Sauvegarder les résultats
            export_manager = ExportManager()
            export_manager.export_to_excel(results, filepath)
            pbar.update(1)
        
        print_success(f"Résultats sauvegardés dans {filepath}")
        return True
    except Exception as e:
        print_error(f"Erreur lors de la sauvegarde : {str(e)}")
        logger.error(f"Erreur détaillée : {traceback.format_exc()}")
        return False

def analyze_results(file_path: str) -> bool:
    """Analyse les résultats et génère un rapport"""
    try:
        logger.info(f"Starting analysis for file: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"Le fichier {file_path} n'existe pas")
            print_error(f"Le fichier {file_path} n'existe pas.")
            return False
            
        print_info("Initialisation de l'analyse...")
        logger.info("Initializing analyzer...")
        
        try:
            analyzer = LitReviewAnalyzer()
            logger.debug("Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'analyseur: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print_error("Impossible d'initialiser l'analyseur.")
            return False
            
        print_info("Chargement des données...")
        logger.info(f"Loading data from {file_path}")
        try:
            load_success = analyzer.load_data(file_path)
            logger.debug(f"Data load result: {'Success' if load_success else 'Failed'}")
            if not load_success:
                print_error("Impossible de charger les données pour l'analyse.")
                return False
                
            # Log DataFrame info after loading
            if analyzer.data is not None:
                logger.debug(f"DataFrame info - Shape: {analyzer.data.shape}, Columns: {analyzer.data.columns.tolist()}")
                logger.debug(f"DataFrame null counts: {analyzer.data.isnull().sum().to_dict()}")
            else:
                logger.error("DataFrame is None after loading")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print_error(f"Erreur lors du chargement des données: {str(e)}")
            return False
            
        # Créer le dossier d'analyse
        try:
            output_dir = os.path.join(os.path.dirname(file_path), "analysis", 
                                    os.path.splitext(os.path.basename(file_path))[0])
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Analysis directory created: {output_dir}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du dossier d'analyse: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print_error("Impossible de créer le dossier d'analyse.")
            return False
            
        print_info("Génération du rapport d'analyse...")
        logger.info("Starting report generation...")
        try:
            # Log DataFrame state before export
            if analyzer.data is not None:
                logger.debug(f"Pre-export DataFrame info - Shape: {analyzer.data.shape}")
                logger.debug(f"Pre-export columns with null values: {analyzer.data.columns[analyzer.data.isnull().any()].tolist()}")
            
            report_path = analyzer.export_analysis(output_dir)
            logger.debug(f"Report generation result: {'Success' if report_path else 'Failed'}")
            
            if not report_path:
                print_error("Échec de la génération du rapport.")
                return False
                
            print_success(f"Rapport généré avec succès: {report_path}")
            
            # Tenter d'ouvrir le rapport dans le navigateur
            try:
                import webbrowser
                webbrowser.open(f'file://{report_path}')
                print_info("Ouverture du rapport dans votre navigateur...")
                logger.info("Report opened in browser")
            except Exception as e:
                logger.error(f"Erreur lors de l'ouverture du rapport: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                print_info(f"Le rapport est disponible ici: {report_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print_error(f"Erreur lors de la génération du rapport: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        print_error(f"Une erreur inattendue s'est produite: {str(e)}")
        return False

def main():
    """Point d'entrée principal"""
    try:
        while True:
            # Afficher le menu
            print("\n" + "=" * 109)
            print(f"{Fore.CYAN}{Style.BRIGHT}" + " " * 29 + "LitReview - Outil d'analyse d'articles scientifiques" + " " * 28 + f"{Style.RESET_ALL}")
            print("=" * 109 + "\n")
            
            print(f"{Fore.WHITE}[1] Rechercher des articles")
            print("[2] Analyser des résultats")
            print(f"[3] Quitter{Style.RESET_ALL}\n")

            try:
                choix = input(f"{Fore.YELLOW}Votre choix (1-3): {Style.RESET_ALL}")
                if choix not in ['1', '2', '3']:
                    print_error("Choix invalide. Veuillez entrer 1, 2 ou 3.")
                    continue

                if choix == '3':
                    print_info("Au revoir!")
                    break

                elif choix == '1':
                    try:
                        # Obtenir les paramètres de recherche
                        query = input("Recherche: ")
                        if not query.strip():
                            print_error("La recherche ne peut pas être vide.")
                            continue
                        
                        try:
                            num_results = input("Nombre d'articles (défaut: 100) [100]: ")
                            num_results = int(num_results) if num_results.strip() else 100
                            if num_results <= 0:
                                print_error("Le nombre d'articles doit être positif.")
                                continue
                        except ValueError:
                            print_error("Veuillez entrer un nombre valide.")
                            continue

                        # Effectuer la recherche
                        print_info("Initialisation de la recherche...")
                        results = search_articles(query, num_results)
                        
                        if results:
                            # Sauvegarder les résultats
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"results_{timestamp}.xlsx"
                            file_path = os.path.join("data", filename)
                            
                            print_info("Sauvegarde des résultats...")
                            if save_results(results, filename):
                                print_success(f"{len(results)} articles sauvegardés dans {filename}")
                                
                                # Analyser les résultats
                                analyze_results(file_path)

                    except KeyboardInterrupt:
                        print_info("Recherche annulée.")
                        continue

                elif choix == '2':
                    try:
                        # Afficher les fichiers de résultats disponibles
                        data_dir = "data"
                        if not os.path.exists(data_dir):
                            print_error("Aucun fichier de résultats trouvé.")
                            continue
                            
                        results_files = [f for f in os.listdir(data_dir) 
                                       if f.endswith('.xlsx') and f.startswith('results_')]
                        
                        if not results_files:
                            print_error("Aucun fichier de résultats trouvé.")
                            continue
                            
                        print_info("Fichiers de résultats disponibles:")
                        for i, file in enumerate(results_files, 1):
                            print(f"{i}. {file}")
                            
                        try:
                            choice = input("\nChoisissez un fichier (numéro): ")
                            file_idx = int(choice) - 1
                            if file_idx < 0 or file_idx >= len(results_files):
                                print_error("Choix invalide.")
                                continue
                                
                            file_path = os.path.join(data_dir, results_files[file_idx])
                            analyze_results(file_path)
                            
                        except ValueError:
                            print_error("Veuillez entrer un numéro valide.")
                            continue
                            
                    except KeyboardInterrupt:
                        print_info("Analyse annulée.")
                        continue

            except KeyboardInterrupt:
                print_info("Opération annulée.")
                continue

    except KeyboardInterrupt:
        print_info("Au revoir !")
    except Exception as e:
        print_error(f"Erreur fatale : {str(e)}")
        logger.error(f"Erreur détaillée : {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()
