#!/bin/bash

# Define the path to the CSV file
csv_path="data/csv/gs_metho.csv"

# Run the Python script with the CSV file as an argument
pipenv run python processor.py --csv $csv_path