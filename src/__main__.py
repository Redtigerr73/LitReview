"""
Main entry point for LitReview CLI
"""

from .cli import LitReviewCLI
from .utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    """
    Main entry point for the application
    """
    logger.info("Starting LitReview CLI...")
    try:
        cli = LitReviewCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("LitReview CLI terminated")

if __name__ == "__main__":
    main()
