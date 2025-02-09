# LitReview

LitReview is an enhanced tool for collecting and analyzing scientific articles from Google Scholar. It provides an interactive interface for searching articles and exports comprehensive metadata in both Excel and CSV formats.

## Features

- Interactive command-line interface
- Progress tracking during article collection
- Enhanced metadata extraction (DOI, abstract, citations)
- Export to Excel with automatic formatting
- Configurable search parameters
- Robust error handling and logging

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/luumsk/LitReview.git
   cd LitReview
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the interactive CLI:
   ```bash
   python -m src.cli
   ```

2. Follow the prompts to:
   - Enter your search query
   - Specify the number of articles to collect (10-1000)
   - Choose export format (Excel/CSV)

3. Results will be saved in the `data` directory with filenames based on:
   - DOI (if available)
   - Search query and timestamp

## Configuration

Edit `config/settings.yaml` to customize:
- Output directory
- Maximum results limit
- Export format
- Rate limiting
- Log level
