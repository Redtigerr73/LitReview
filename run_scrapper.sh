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

# Run the Python script with the list of queries
python scraper.py "${queries[@]}"