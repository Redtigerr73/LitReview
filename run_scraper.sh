#!/bin/bash

# Define the list of queries
queries=(
    "multiple sclerosis segmentation deep learning"
    "multiple sclerosis segmentation U-net"
    "multiple sclerosis segmentation deep learning review"
    "multiple sclerosis segmentation gan"
    "multiple sclerosis segmentation cnn"
    "multiple sclerosis segmentation deep learning 3D"
    "deep learning models for multiple sclerosis segmentation"
)
# queries=(
#     "multiple sclerosis"
#     "multiple sclerosis brain mri"
#     "multiple sclerosis types and characteristics"
#     "multiple sclerosis Russia Siberia"
#     "multiple sclerosis diagnosis"
#     "multiple sclerosis cost Russia"
#     "multiple sclerosis statistics"
# )
max_page=7
start_num=0
outfile="data/csv/gs_articles2.csv"

# Run the Python script with the list of queries
pipenv run python scraper.py "${queries[@]}" -m $max_page -o $outfile -s $start_num