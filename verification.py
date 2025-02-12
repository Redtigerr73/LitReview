import os
from pathlib import Path

def verify_environment():
    required = {
        'drivers/chromedriver.exe': 'ChromeDriver',
        'config/settings.yaml': 'Configuration'
    }
    
    missing = []
    for path, name in required.items():
        if not Path(path).exists():
            missing.append(name)
    
    if missing:
        print("Éléments manquants :")
        print("\n".join(missing))
        print("\nExécutez :")
        print("python download_chromedriver.py")
    else:
        print("Environnement OK")

if __name__ == "__main__":
    verify_environment()
