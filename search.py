import time
import random
import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup


def wait():
    print("Waiting for 5 seconds...")
    time.sleep(5)

def process_query(query):
    processed = str(query).strip().lower().replace(" ", "+")
    return processed

def set_up_url(query):
    query_str = process_query(query)
    return f"https://scholar.google.com/scholar?hl=en&q={query_str}"

def update_url(url, start_num):
    return f"{url}&start={start_num}"

def save_response_to_html(url, filename='response.html', headers=None):
    response = requests.get(url, headers=headers)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(response.text)

def read_html_to_beautifulsoup(filename='response.html'):
    with open(filename, 'r', encoding='utf-8') as file:
        html_content = file.read()
    return html_content

def scrap(url):
    # response = requests.get(url, headers=requests.utils.default_headers())
    # response.raise_for_status()
    # return response
    return read_html_to_beautifulsoup(filename='response.html')
    #save_response_to_html(url, filename='response.html', headers=requests.utils.default_headers())
    

def get_statistics(response):
    print("Total page number")
    return {"n_page_total": 9999}

def extract_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    results = soup.find("div", id="gs_res_ccl_mid")
    job_elements = results.find_all("div", class_="gs_ri")
    
    for job_element in job_elements:
        # Refenrece
        ref_element = job_element.find("div", class_="gs_a")
        ref = ref_element.text
        splits = ref.split("-")

        # Author
        authors = " ,".join(splits[:-2]).replace(r"\xa0", "")

        # Publisher
        publisher = splits[-1].strip()

        # Year of publishment
        year = int(splits[-2].strip()[-4:])

        # Link to paper
        links = job_element.find("a")
        link_url = links["href"]

        # Title of the paper
        title_element = links.text.strip()

        citation = job_element.find("div", class_="gs_fl gs_flb")
        n_cit = citation.find_all("a")
        n_cit = [int(a.text.split(" ")[-1]) for a in n_cit if "Cited by" in a.text][0]
        
        print(f"Title: {title_element}")
        print(f"Url: {link_url}")
        print(f"Authors: {authors}")
        print(f"Year: {year}")
        print(f"Publisher: {publisher}")
        print(f"Cite by: {n_cit}", end="\n\n")

def append(data, outfile): print(f"Appended data to {outfile}")


def main(query: str, outfile:str="articles.csv"):
    url = set_up_url(query)
    print(f"URL: {url}")

    try:
        response = scrap(url)
        # for i in range(1, n_page_to_scrap):
        #     print(f"Scrapping page {i}")
        #     start_num = 10 * i
        #     url = update_url(url, start_num)
        #     response = scrap(url)
        #     wait()

        #     if response.status_code != 200:
        #         print(f"ERROR: {response}")
        #         break
        #     else:
        #         data = extract_data(response)
        #         append(data, outfile)
    except HTTPError as err:
        print(err)
    
    
    # statistics = get_statistics(response)
    # n_page_to_scrap = min(statistics["n_page_total"], 2)
    data = extract_data(response)
    # append(data, outfile)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="query")
    args = parser.parse_args()
    main(query=args.query, outfile="test.csv")








