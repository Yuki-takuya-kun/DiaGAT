# Dialogue Graph Attention Network in
Conversational Aspect-base Sentiment Analysis

This repository contains data and code for the paper: Dialogue Graph Attention Network in Conversational Aspect-base Sentiment Analysis

## Directory structure
<pre>
DiaGAT					root directory
├── data				data directory
│     └── dataset			dataset directory
│           ├──jsons_en			english dataset directory
│           │	├──train.json		english training dataset
│           │	├──valid.json		english validation dataset
│           │	└──test.json		english test dataset
│           └──jsons_zh			chinese dataset directory
│        	├──train.json		english training dataset
│             	├──valid.json		english validation dataset
│      	       	└──test.json		english test dataset
└── src					source code of the DiaGAT model
      ├── common.py
      ├── preprocess.py			encode the dialogue into dialogue graph
      ├── dataloader.py			load the dialogue graph into dataset
      ├── model.py			define architecture of the DiaGAT model
      ├── metrics.py			evaluate the behavior of the model
      └── config.yaml			configuration file to setting many hyper parameters 			
	
</pre>

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
