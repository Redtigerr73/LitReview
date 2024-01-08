import time
import csv
import random
import os
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
    return f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={query_str}"

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
    data = []

    soup = BeautifulSoup(html_content, 'html.parser')
    results = soup.find("div", id="gs_res_ccl_mid")
    job_elements = results.find_all("div", class_="gs_ri")
    
    for job_element in job_elements:
        ref_element = job_element.find("div", class_="gs_a").text
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
        n_cit = [int(a.text.split(" ")[-1]) for a in citation_elements if "Cited by" in a.text][0]

        data.append({
            "title"       : title,
            "url"         : url,
            "first_author": first_author,
            "authors"     : authors_str,
            "year"        : year,
            "publisher"   : publisher,
            "cited_by"    : n_cit,
        })

    return data

def append_to_csv(file_path, data_list):
    file_exists = os.path.isfile(file_path)

    # Open the CSV file in append mode
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data_list[0].keys())

        # Write header only if the file is empty
        if not file_exists:
            writer.writeheader()

        # Write data rows
        writer.writerows(data_list)
    print(f"Append data to file {file_path}")

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
    append_to_csv(outfile, data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="query")
    args = parser.parse_args()
    main(query=args.query, outfile="test.csv")








