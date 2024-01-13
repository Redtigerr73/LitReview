#!/bin/bash

# Define the list of queries
queries=(
    # "multiple sclerosis segmentation deep learning"
    # "multiple sclerosis segmentation U-net"
    # "multiple sclerosis segmentation GAN"
    # "multiple sclerosis segmentation CNN"
    # "multiple sclerosis segmentation nnU-net"
    # "deep learning models for multiple sclerosis segmentation"
    "multiple sclerosis MRI segmentation"
    "small brain tissue segmentation deep learning"
    "brain leisions segmentation deep learning"
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
max_page=5
start_num=0
outfile="data/csv/gs_ms_seg.csv"

# Run the Python script with the list of queries
pipenv run python scraper.py "${queries[@]}" -m $max_page -o $outfile -s $start_num