# DiaGAT
This repository contains data and code for the paper: Dialogue Graph Attention Network in Conversational Aspect-base Sentiment Analysis
## Requirements:
Our model needs packages to run:
- pytorch >= 1.12.1
- torch-lightning >= 2.0.0

## Quick Start
- First you are trainning the model in the first time, you need to initialize the dataset by
`python main.py -d True`
- Or you can trainning the model directly
`python main.py`

## Configuration
If you want to change configurations in the model, you can modify the configuration file `src/config` to change config such as language