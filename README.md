# LitReview

LitReview is a program that collect scientific articles from Google Scholar based on a query string, extract information from the found articles and save to a CSV file.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/luumsk/LitReview.git
    ```
2. Navigate to the project directory:
    ```bash
    cd LitReview
    ```
3. Install `pipenv` if it's not already installed:
    ```bash
    pip install pipenv
    ```
4. Install the required Python packages from the `Pipfile`:
    ```bash
    pipenv install
    ```

### Usage

You can run the scraper with a list of queries as follows:

```bash
python scraper.py "query1" "query2" "query3"
