import time
import random

class Response():
    def __init__(self) -> None:
        self.status_code = 429

    def __str__(self):
        return f"<Response {self.status_code}>"


def wait():
    print("Waiting for 5 seconds...")
    time.sleep(5)

def process_query(query): print(f"--> Process query {query}")
def set_up_url(query_str): print("--> Set up url")
def update_url(url, start_num): print("Update url")
def scrap(url):
    print("--> Scrap")
    return Response()
def get_statistics(response):
    print("Total page number")
    return {"n_page_total": 9999}
def extract_data(reponse): print("--> Extract data")
def append(data, outfile): print(f"Appended data to {outfile}")
def handle_error(status_code):
    if status_code == 429:
        print(f"HTTP CODE {status_code}: You are sending too many requests. Try again another time.")

def main(query: str, outfile:str="articles.csv"):
    query_str = process_query(query)
    url = set_up_url(query_str)
    print(f"Scrapping page 0")
    response = scrap(url)

    if response.status_code != 200:
        handle_error(response.status_code)
    else:
        statistics = get_statistics(response)
        n_page_to_scrap = min(statistics["n_page_total"], 2)
        data = extract_data(response)
        append(data, outfile)

        for i in range(1, n_page_to_scrap):
            print(f"Scrapping page {i}")
            start_num = 10 * i
            url = update_url(url, start_num)
            response = scrap(url)
            wait()

            if response.status_code != 200:
                print(f"ERROR: {response}")
                break
            else:
                data = extract_data(response)
                append(data, outfile)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="query")
    args = parser.parse_args()
    main(query=args.query)








