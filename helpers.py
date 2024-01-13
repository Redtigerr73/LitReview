import time
import csv
import random
import requests

def wait(seconds=None):
    """
    Pauses the execution of the program for a specified number of seconds.
    If no number is specified, the pause will last for a random number of seconds between 1 and 5.

    Parameters:
    seconds (int, optional): The number of seconds to pause. Defaults to a random number between 1 and 5.

    Returns:
    None
    """
    if not seconds:
        seconds = random.randint(1, 5)
    print(f"Waiting for {seconds} seconds...")
    time.sleep(seconds)

def append_to_csv(csv_path, data_list):
    """
    Appends a list of dictionaries to a CSV file.

    Parameters:
    csv_path (str): The path to the CSV file.
    data_list (list): The list of dictionaries to append.

    Returns:
    None
    """
    if len(data_list) == 0:
        print("No data to append")
        return

    # Get the fieldnames from the first dictionary in data_list
    fieldnames = data_list[0].keys()

    # Open the CSV file in append mode
    with open(csv_path, 'a+', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Move to the start of the file
        file.seek(0)

        # Check if the file is empty
        if file.read(1) == '':
            writer.writeheader()
        else:
            # Move to the start of the file again
            file.seek(0)

            # Read the header from the CSV file
            existing_header = next(csv.reader(file))

            # Check if the header of the CSV file is the same as the fieldnames
            if existing_header != list(fieldnames):
                print("Cannot append to CSV file. The headers of the CSV file do not match the keys of the dictionaries in data_list")

                return

        # Write data rows
        writer.writerows(data_list)

    print(f"Appended data to file {csv_path}")

def fetch_webpage(url):
    """
    Fetches the webpage at the specified URL.

    Parameters:
    url (str): The URL of the webpage to fetch.

    Returns:
    requests.Response: The response from the server.
    """

    print(f"URL: {url}")
    response = requests.get(url, headers=requests.utils.default_headers())
    response.raise_for_status()
    return response