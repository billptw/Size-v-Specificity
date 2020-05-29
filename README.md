# Size v. Specificity: Comparing Pre-trained Embeddings for Disaster Tweet Classification

## Introduction
With the proliferation of social media, Twitter has become an important channel for emergency respondents to conduct automated monitoring of on-the-ground updates on disasters. To this end, Disaster Tweet Classification is a motivated task to identify if tweets are referencing disasters or not. We begin with training an LSTM encoder to learn the word embeddings from a corpus built using the "Real or Not? NLP with Disaster Tweets" classification dataset, with a final linear classifier layer for the task. However, the success of this model is limited, as the disaster tweet classification dataset available is small. With the advent of pre-trained models in NLP that achieves state-of-the-art (SOTA) results on many tasks, we sought to leverage pre-trained embeddings as inputs to our classifier. In choosing the ideal corpus for pre-training, we observe a trade-off between size and domain-matching; increasing the size of a pre-trained corpus makes the source domain distribution less similar to the target domain. This brings us to the central question: Is it more effective to have a larger but more general pre-trained corpus, or a smaller one that is more similar to the target domain? Our experiments on LSTM with GloVe embeddings and pre-trained BERT show the former, highlighting the importance of large corpora when pre-training for low-resource transfer learning.


## Usage
### Dependencies
This project uses PyTorch version 1.5 with CUDA version 10.1. Install the dependencies via the requirements.txt file by running the following command.
```
pip install -r requirements.txt
```

### Training and Evaluation
To train and evaluate the three different models in the paper, use the following commands below. The models are set to run on the GPUs by default.

* Baseline LSTM model
```
python baseline_LSTM.py --batch_size 32 --lr 0.001 --epochs 20
```
* LSTM with Glove 6B, Twitter 27B or Common Crawl 42B. The glove argument can be 6B, 27B or 42B. After selecting the specific GloVe, it will be downloaded to the current folder you are in.
```
python baseline_LSTM_GloVe.py --batch_size 32 --lr 0.0005 --glove 6B --epochs 20
```

* BERT Model
```
python BERT.py --batch_size 32 --lr 2e-5 --epochs 4
```

## Credits

We thank the HuggingFace team for [BERT](https://github.com/huggingface/transformers) implementation, and Stanford NLP for the [GloVe](https://nlp.stanford.edu/projects/glove/) pre-trained embeddings.

This project was part of the [CE7455](https://ntunlpsg.github.io/ce7455_deep-nlp-20/): Deep Learning for Natural Language Processing final project, done in collaboration with [Clement Tan](https://github.com/txrc). We thank Prof [Shafiq Joty](https://raihanjoty.github.io/) and [team](https://ntunlpsg.github.io/) for their valuable advice.
