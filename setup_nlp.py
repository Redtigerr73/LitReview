"""
Setup script for installing required NLP models
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_spacy_models():
    """Install required spaCy models for English and French"""
    models = [
        "en_core_web_md",  # English model (medium)
        "fr_core_news_md"  # French model (medium)
    ]
    
    logger.info("Installing spaCy models...")
    
    for model in models:
        try:
            logger.info(f"Installing {model}...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
            logger.info(f"Successfully installed {model}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing {model}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error installing {model}: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        install_spacy_models()
        logger.info("All NLP models installed successfully!")
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)
