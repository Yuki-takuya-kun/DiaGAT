# Dialogue Graph Attention Network in Conversational Aspect-base Sentiment Analysis

This repository contains data and code for the paper: Dialogue Graph Attention Network in Conversational Aspect-base Sentiment Analysis

## Overview
In our work, we propose DiaGAT model. The DiaGAT model presents a significant advancement in conversational aspect-based sentiment analysis by effectively leveraging dialogue structure and syntax to improve sentiment analysis accuracy. This model introduces a dual graph attention mechanism, namely Utterance GAT and Token GAT, which are designed to capture the complex dynamics of conversations, particularly in hierarchical relationships and syntactic dependencies.

### Key Features
- __Graph-Based Modeling__: Constructs a comprehensive graph integrating both reply and syntactic relations across dialogues, capturing the intricate relationships between utterances and within utterances.
- __Hierarchical Attention Mechanism__: Utilizes two layers of graph attention networks to process different levels of information flow — one at the utterance level and another at the token level, ensuring a robust understanding of the context.
- __Enhanced Tagging Scheme__: Implements a refined tagging scheme that constrains the prediction space, improving the precision of identifying sentiment-related terms within conversations.

### Performance
DiaGAT outperforms existing models, including state-of-the-art generative models like GPT-4, in accurately extracting sentiment quadruples from conversations. It demonstrates superior capability in handling complex dialogue scenarios across multiple datasets in both English and Chinese.


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
- If you are trainning the model in the first time, you need to initialize the dataset by

`python main.py -d True`
- Or you can trainning the model directly by

`python main.py`

## Configuration
If you want to change hyperparameters in the model, you can modify the configuration file `src/config` to change config such as language
