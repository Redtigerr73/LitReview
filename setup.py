"""
Script d'installation pour LitReview
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_wkhtmltopdf_instructions():
    """Affiche les instructions pour installer wkhtmltopdf manuellement"""
    print("\nInstallation de wkhtmltopdf requise:")
    print("1. Téléchargez wkhtmltopdf depuis: https://wkhtmltopdf.org/downloads.html")
    print("2. Choisissez la version Windows 64-bit")
    print("3. Exécutez l'installateur en tant qu'administrateur")
    print("4. Ajoutez le chemin suivant à votre PATH système:")
    print("   C:\\Program Files\\wkhtmltopdf\\bin")
    print("\nUne fois l'installation terminée, vous pourrez générer des PDF depuis LitReview.")

def setup_virtual_env():
    """Configure l'environnement virtuel et installe les dépendances"""
    print("Configuration de l'environnement virtuel...")
    
    venv_dir = "venv"
    if not os.path.exists(venv_dir):
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    
    # Déterminer le chemin de pip dans l'environnement virtuel
    if platform.system().lower() == "windows":
        pip_path = os.path.join(venv_dir, "Scripts", "pip.exe")
        python_path = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_dir, "bin", "pip")
        python_path = os.path.join(venv_dir, "bin", "python")
    
    # Installer les dépendances
    print("Installation des dépendances...")
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    
    # Installation des modèles et ressources
    print("\nInstallation des modèles et ressources...")
    
    try:
        # spaCy
        print("\nInstallation des modèles spaCy...")
        subprocess.run([python_path, "-m", "spacy", "download", "en_core_web_md"], check=True)
        subprocess.run([python_path, "-m", "spacy", "download", "fr_core_news_md"], check=True)
    except Exception as e:
        print(f"Erreur lors de l'installation des modèles spaCy: {e}")
    
    try:
        # NLTK
        print("\nInstallation des ressources NLTK...")
        subprocess.run([python_path, "-m", "nltk.downloader", "punkt"], check=True)
        subprocess.run([python_path, "-m", "nltk.downloader", "stopwords"], check=True)
        subprocess.run([python_path, "-m", "nltk.downloader", "averaged_perceptron_tagger"], check=True)
    except Exception as e:
        print(f"Erreur lors de l'installation des ressources NLTK: {e}")
    
    try:
        # TextBlob
        print("\nInstallation des ressources TextBlob...")
        subprocess.run([python_path, "-m", "textblob.download_corpora"], check=True)
    except Exception as e:
        print(f"Erreur lors de l'installation des ressources TextBlob: {e}")

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "logs",
        "output",
        "output/pdf",
        "output/excel",
        "drivers"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Répertoire créé/vérifié: {directory}")

def main():
    """Fonction principale d'installation"""
    print("Démarrage de l'installation de LitReview...")
    
    try:
        # Créer les répertoires
        create_directories()
        
        # Configurer l'environnement virtuel et installer les dépendances
        setup_virtual_env()
        
        # Afficher les instructions pour wkhtmltopdf
        print_wkhtmltopdf_instructions()
        
        print("\nInstallation des composants Python terminée avec succès!")
        print("\nPour commencer à utiliser LitReview:")
        print("1. Activez l'environnement virtuel:")
        if platform.system().lower() == "windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("2. Installez wkhtmltopdf en suivant les instructions ci-dessus")
        print("3. Lancez le programme:")
        print("   python litreview.py")
        
    except Exception as e:
        print(f"\nErreur lors de l'installation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
