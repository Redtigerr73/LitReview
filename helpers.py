import time
import csv

def wait():
    print("Waiting for 5 seconds...")
    time.sleep(5)

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