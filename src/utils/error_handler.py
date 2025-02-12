"""Module de gestion des erreurs et exceptions"""

import logging
import traceback
from typing import Optional, Dict, Any
from functools import wraps
import sys
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
    SessionNotCreatedException
)

class ScraperException(Exception):
    """Exception de base pour les erreurs de scraping"""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class BrowserInitError(ScraperException):
    """Erreur d'initialisation du navigateur"""
    pass

class CaptchaError(ScraperException):
    """Erreur liée au CAPTCHA"""
    pass

class NetworkError(ScraperException):
    """Erreur de réseau"""
    pass

class ParsingError(ScraperException):
    """Erreur de parsing des données"""
    pass

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Configure un logger avec gestion des fichiers et de la console"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter les handlers en double
    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler fichier si spécifié
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def handle_selenium_exceptions(func):
    """Décorateur pour gérer les exceptions Selenium"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SessionNotCreatedException as e:
            raise BrowserInitError(
                "Erreur d'initialisation du navigateur Chrome",
                "CHROME_INIT_ERROR",
                {"original_error": str(e)}
            )
        except TimeoutException as e:
            raise NetworkError(
                "Délai d'attente dépassé",
                "TIMEOUT_ERROR",
                {"original_error": str(e)}
            )
        except WebDriverException as e:
            if "chrome not reachable" in str(e).lower():
                raise BrowserInitError(
                    "Chrome n'est pas accessible",
                    "CHROME_NOT_REACHABLE",
                    {"original_error": str(e)}
                )
            raise ScraperException(
                f"Erreur WebDriver: {str(e)}",
                "WEBDRIVER_ERROR",
                {"original_error": str(e)}
            )
        except Exception as e:
            raise ScraperException(
                f"Erreur inattendue: {str(e)}",
                "UNEXPECTED_ERROR",
                {
                    "original_error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
    return wrapper

def retry_on_exception(retries: int = 3, delay: int = 1):
    """Décorateur pour réessayer une opération en cas d'erreur"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            last_exception = None
            
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay * (attempt + 1))  # Délai exponentiel
                        continue
                    raise last_exception
        return wrapper
    return decorator

class ErrorTracker:
    """Classe pour suivre et analyser les erreurs"""
    
    def __init__(self):
        self.errors = []
        self.logger = setup_logger("error_tracker")
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Enregistre une erreur avec son contexte"""
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        self.errors.append(error_info)
        self.logger.error(
            f"Error tracked: {error_info['type']} - {error_info['message']}",
            extra={'context': context}
        )
    
    def get_error_summary(self) -> Dict[str, int]:
        """Retourne un résumé des erreurs par type"""
        from collections import Counter
        return Counter(error['type'] for error in self.errors)
    
    def clear_errors(self):
        """Efface l'historique des erreurs"""
        self.errors.clear()
    
    def has_critical_errors(self) -> bool:
        """Vérifie s'il y a des erreurs critiques"""
        critical_types = {
            'BrowserInitError',
            'CaptchaError',
            'NetworkError'
        }
        return any(error['type'] in critical_types for error in self.errors)

# Instance globale du tracker d'erreurs
error_tracker = ErrorTracker()

def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """Fonction utilitaire pour logger une erreur avec contexte"""
    error_tracker.track_error(error, context)
    logger.error(
        f"Error occurred: {type(error).__name__} - {str(error)}",
        exc_info=True,
        extra={'context': context}
    )
