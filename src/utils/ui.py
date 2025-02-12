"""Module pour l'interface utilisateur avec des couleurs et du style"""
import os
import sys
import threading
from colorama import init, Fore, Back, Style
from typing import List, Optional
import time

# Initialize colorama for Windows
init()

class ProgressSpinner:
    """Spinner animé pour montrer une progression"""
    def __init__(self, message: str = ""):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current = 0
        self.message = message
        
    def spin(self):
        """Affiche le prochain caractère du spinner"""
        sys.stdout.write(f"\r{Fore.CYAN}{self.spinner_chars[self.current]} {self.message}{Style.RESET_ALL}")
        sys.stdout.flush()
        self.current = (self.current + 1) % len(self.spinner_chars)

class SpinnerContext:
    """Context manager pour le spinner"""
    def __init__(self, message: str = ""):
        self.spinner = ProgressSpinner(message)
        self.stop = False
        self.thread = None
        
    def spin_worker(self):
        """Worker thread pour faire tourner le spinner"""
        while not self.stop:
            self.spinner.spin()
            time.sleep(0.1)
        
    def __enter__(self):
        self.thread = threading.Thread(target=self.spin_worker)
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = True
        if self.thread:
            self.thread.join()
        print()

def wait_with_spinner(message: str = "") -> SpinnerContext:
    """Retourne un context manager pour le spinner"""
    return SpinnerContext(message)

def clear_screen():
    """Efface l'écran"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    """Affiche un titre stylisé"""
    width = os.get_terminal_size().columns
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * width}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{title.center(width)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * width}{Style.RESET_ALL}\n")

def print_menu(options: List[str]) -> None:
    """Affiche un menu stylisé"""
    for i, option in enumerate(options, 1):
        print(f"{Fore.GREEN}{Style.BRIGHT}[{i}]{Style.RESET_ALL} {option}")
    print()

def print_success(message: str):
    """Affiche un message de succès"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}✓ {message}{Style.RESET_ALL}")

def print_error(message: str):
    """Affiche un message d'erreur"""
    print(f"\n{Fore.RED}{Style.BRIGHT}✗ {message}{Style.RESET_ALL}")

def print_warning(message: str):
    """Affiche un avertissement"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}⚠ {message}{Style.RESET_ALL}")

def print_info(message: str):
    """Affiche une information"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}ℹ {message}{Style.RESET_ALL}")

def print_progress(current: int, total: int, prefix: str = "", suffix: str = ""):
    """Affiche une barre de progression"""
    percent = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="\r")
    if current == total:
        print()

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Obtient une entrée utilisateur avec un style cohérent"""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    return input(f"{Fore.GREEN}{Style.BRIGHT}{prompt}{Style.RESET_ALL}").strip() or default or ""

def get_user_choice(min_value: int, max_value: int, prompt: str = "Votre choix") -> int:
    """Obtient un choix numérique de l'utilisateur"""
    while True:
        try:
            choice = get_user_input(f"{prompt} ({min_value}-{max_value})")
            value = int(choice)
            if min_value <= value <= max_value:
                return value
            print_error(f"Veuillez entrer un nombre entre {min_value} et {max_value}")
        except ValueError:
            print_error(f"Veuillez entrer un nombre valide entre {min_value} et {max_value}")

def get_yes_no(prompt: str, default: Optional[bool] = None) -> bool:
    """Obtient une réponse oui/non de l'utilisateur"""
    if default is not None:
        prompt = f"{prompt} [{'O/n' if default else 'o/N'}]"
    else:
        prompt = f"{prompt} [o/n]"
        
    while True:
        choice = get_user_input(prompt).lower()
        if not choice and default is not None:
            return default
        elif choice in ["o", "oui", "yes", "y"]:
            return True
        elif choice in ["n", "non", "no"]:
            return False
        print_error("Veuillez répondre par 'o' ou 'n'")
