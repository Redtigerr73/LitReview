"""
Script pour télécharger ChromeDriver avec gestion détaillée des erreurs
"""
import os
import sys
import requests
import zipfile
import logging
import traceback
from pathlib import Path
import shutil
import stat
import platform
import subprocess
import re

def setup_logging():
    """Configure detailed logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "chromedriver_download.log"
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def get_chrome_version_windows():
    """Get Chrome version on Windows using registry or file version"""
    logger = logging.getLogger(__name__)
    
    chrome_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    
    for path in chrome_paths:
        if os.path.exists(path):
            logger.info(f"Chrome found at: {path}")
            try:
                # Use PowerShell to get file version info
                escaped_path = path.replace('"', '\\"')
                cmd = f'powershell -command "(Get-Item \'{escaped_path}\').VersionInfo.FileVersion"'
                result = subprocess.check_output(cmd, shell=True).decode().strip()
                
                # Extract version number using regex
                version_match = re.search(r'\d+\.\d+\.\d+\.\d+', result)
                if version_match:
                    version = version_match.group(0)
                    logger.info(f"Chrome version: {version}")
                    return version
            except Exception as e:
                logger.error(f"Error getting Chrome version: {e}")
                logger.debug(traceback.format_exc())
    
    logger.error("Chrome not found in standard locations")
    raise FileNotFoundError("Chrome is not installed in standard locations")

def verify_chrome_installation():
    """Verify Chrome installation and get version"""
    logger = logging.getLogger(__name__)
    
    try:
        if platform.system().lower() == "windows":
            return get_chrome_version_windows()
        else:
            logger.error("This script is designed for Windows only")
            raise OSError("This script is designed for Windows only")
            
    except Exception as e:
        logger.error(f"Error verifying Chrome installation: {e}")
        logger.debug(traceback.format_exc())
        raise

def make_executable(path):
    """Make a file executable"""
    logger = logging.getLogger(__name__)
    try:
        # Add executable bit (chmod +x)
        current = stat.S_IMODE(os.lstat(path).st_mode)
        os.chmod(path, current | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        logger.debug(f"Made {path} executable")
    except Exception as e:
        logger.error(f"Error making file executable: {e}")
        logger.debug(traceback.format_exc())
        raise

def verify_downloaded_driver(driver_path):
    """Verify the downloaded ChromeDriver"""
    logger = logging.getLogger(__name__)
    
    try:
        if not os.path.exists(driver_path):
            raise FileNotFoundError(f"ChromeDriver not found at {driver_path}")
        
        # Check file size
        size = os.path.getsize(driver_path)
        if size < 1000000:  # Less than 1MB
            raise ValueError(f"ChromeDriver file seems too small ({size} bytes)")
        
        # Make executable
        make_executable(driver_path)
        
        # Try to execute with --version
        try:
            result = subprocess.run([driver_path, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            logger.info(f"ChromeDriver version check: {result.stdout.strip()}")
        except Exception as e:
            logger.error(f"Error checking ChromeDriver version: {e}")
            logger.debug(traceback.format_exc())
            raise
            
    except Exception as e:
        logger.error(f"Error verifying ChromeDriver: {e}")
        logger.debug(traceback.format_exc())
        raise

def download_chromedriver():
    """Download ChromeDriver with comprehensive error handling"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create drivers directory if it doesn't exist
        driver_dir = Path(__file__).parent / "drivers"
        driver_dir.mkdir(exist_ok=True)
        logger.info(f"Using driver directory: {driver_dir}")
        
        chromedriver_path = driver_dir / "chromedriver.exe"
        
        # Verify Chrome installation first
        chrome_version = verify_chrome_installation()
        major_version = chrome_version.split('.')[0]
        logger.info(f"Detected Chrome major version: {major_version}")
        
        # Direct download URL for ChromeDriver
        version = "133.0.6943.53"  # Match with Chrome 133
        download_url = f"https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/{version}/win64/chromedriver-win64.zip"
        logger.info(f"Download URL: {download_url}")
        
        # Download ChromeDriver
        logger.info("Downloading ChromeDriver...")
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save and extract ChromeDriver
        zip_path = driver_dir / "chromedriver.zip"
        logger.debug(f"Saving zip to: {zip_path}")
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file
        logger.info("Extracting ChromeDriver...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract chromedriver.exe from the correct path in the zip
            for file in zip_ref.namelist():
                if file.endswith('chromedriver.exe'):
                    zip_ref.extract(file, driver_dir)
                    extracted_path = driver_dir / file
                    if os.path.exists(chromedriver_path):
                        logger.debug(f"Removing existing ChromeDriver: {chromedriver_path}")
                        os.remove(chromedriver_path)
                    logger.debug(f"Moving {extracted_path} to {chromedriver_path}")
                    os.rename(extracted_path, chromedriver_path)
                    break
        
        # Clean up
        logger.debug("Cleaning up temporary files...")
        os.remove(zip_path)
        
        # Remove any leftover directories
        for item in driver_dir.iterdir():
            if item.is_dir():
                logger.debug(f"Removing directory: {item}")
                shutil.rmtree(item)
        
        # Verify the downloaded driver
        logger.info("Verifying downloaded ChromeDriver...")
        verify_downloaded_driver(chromedriver_path)
        
        logger.info(f"ChromeDriver successfully downloaded and verified at: {chromedriver_path}")
        return str(chromedriver_path)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading ChromeDriver: {e}")
        logger.debug(traceback.format_exc())
        raise
    except zipfile.BadZipFile as e:
        logger.error(f"Error extracting ChromeDriver zip: {e}")
        logger.debug(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading ChromeDriver: {e}")
        logger.debug(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting ChromeDriver download process...")
        
        driver_path = download_chromedriver()
        print(f"\nChromeDriver downloaded successfully to: {driver_path}")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nError: {e}")
        sys.exit(1)