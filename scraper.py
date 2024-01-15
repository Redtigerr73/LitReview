import logging
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
from helpers import append_to_csv, wait, fetch_webpage

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')


class ScholarScraper:
    """
    A class used to scrape data from Google Scholar.
    ...

    Attributes
    ----------
    query : str
        the query to search for on Google Scholar
    outfile : str
        the path to the output CSV file

    Methods
    -------
    run(query)
        Runs the scraper for a specific query.
    """
    def __init__(self, outfile):
        self.query   = None
        self.outfile = outfile

    def _set_up_url(self, start_num=0):
        """
        Sets up the URL for the Google Scholar search.

        Parameters:
        start_num (int): The start number for the search results.

        Returns:
        str: The URL for the Google Scholar search.
        """

        if not self.query:
            raise ValueError("Query is not set")
        if not isinstance(self.query, str):
            raise ValueError("Query is not a string")
        
        processed_query = self.query.strip().lower().replace(" ", "+")
        return f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={processed_query}&start={start_num}"


    def _extract_article_info(self, job_element):
        """
        Extracts the article information from a job element.

        Parameters:
        job_element (bs4.element.Tag): The job element to extract the article information from.

        Returns:
        dict: The article information.
        """

        ref_element = job_element.find("div", class_="gs_a").text
        split_ref   = ref_element.split("-")

        if len(split_ref) < 3:
            return None
        
        *authors, ref_text, publisher = ref_element.split("-")

        # Author
        authors_str = " ,".join(authors).replace(u"\xa0", u"")
        first_author = authors_str.split(",")[0]

        # Publisher
        publisher = publisher.strip()

        # Year of publishment is the last 4 characters of ref_text
        year = int(ref_text.strip()[-4:])

        # Link to paper
        link = job_element.find("a")
        url = link["href"]

        # Title of the paper
        title = link.text.strip()

        # Number of citation
        citation_elements = job_element.find("div", class_="gs_fl gs_flb").find_all("a")
        n_cit_elements = [a for a in citation_elements if "Cited" in a.text]
        n_cit = int(n_cit_elements[0].text.split(" ")[-1]) if n_cit_elements else 0

        return {
            "title"       : title,
            "url"         : url,
            "year"        : year,
            "cited_by"    : n_cit,
            "publisher"   : publisher,
            "first_author": first_author,
            "authors"     : authors_str,
        }

    def _extract_articles_from_html(self, html_content):
        """
        Extracts the articles from the HTML content.

        Parameters:
        html_content (str): The HTML content to extract the articles from.

        Returns:
        list: A list of dictionaries containing the article information.
        """

        article_list = []
        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find("div", id="gs_res_ccl_mid")

        if divs:
            job_elements = divs.find_all("div", class_="gs_ri")
            for job_element in job_elements:
                try:
                    article_info = self._extract_article_info(job_element)
                    article_list.append(article_info)
                except: # skip the current article if there's an error
                    continue

        return article_list

    def process_page(self, start):
        """
        Processes a page of search results.

        Parameters:
        start (int): The start number for the search results.

        Returns:
        None
        """

        try:
            url = self._set_up_url(start)
            response = fetch_webpage(url)
            article_list = self._extract_articles_from_html(response.text)

            if len(article_list) == 0:
                return

            for article in article_list:
                if not article:
                    return
                article["query"] = self.query

            append_to_csv(self.outfile, article_list)
            wait()
        except HTTPError as err:
            logging.error("Error while fetching data: %s", err)
            raise
        except Exception as err:
            logging.error("Error while extracting data: %s", err)
            raise

    def run(self, queries, max_page, start_num):
        """
        Runs the scraper for a list of queries.

        Parameters:
        queries (list): The list of queries to search for on Google Scholar.
        max_page (int): The maximum number of pages to scrape for each query.
        start_num (int): The starting page number for each query.

        Returns:
        None
        """
        for query in queries:
            logging.info(f"Processing query: '{query}'. Start={start_num}, Max={max_page}")
            self.query = query
            page_size  = 10
            current_start_num = start_num

            for i in range(max_page):
                logging.info(f"Page {i}")
                self.process_page(current_start_num)
                current_start_num += page_size
                print("")
            print("--------------------")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="queries", nargs='+')
    parser.add_argument(
        "-o", "--outfile",
        dest="outfile",
        default="gs_articles.csv"
    )
    parser.add_argument(
        "-m", "--maxPage",
        dest="max_page",
        default=5,
        type=int
    )
    parser.add_argument(
        "-s", "--startNum",
        dest="start_num",
        default=0,
        type=int
    )
    args = parser.parse_args()
    scraper = ScholarScraper(outfile=args.outfile)
    scraper.run(
        queries=args.queries,
        max_page=args.max_page,
        start_num=args.start_num
    )
