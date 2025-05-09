# language-detection

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fasttext
pip install numpy==1.26.4
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
wget https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual/resolve/main/dataset/es/es_1000.jsonl
wget https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual/resolve/main/dataset/en/en_1000.jsonl
wget https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual/resolve/main/dataset/gr/gr_1000.jsonl
wget https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual/resolve/main/dataset/ja/ja_1000.jsonl
mv gr_1000.jsonl el_1000.jsonl

```

## Ideas

- Make a cli chatbot based on the example of the course
- Turn it into a discord bot
- Test the code on [a real dataset](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multilingual)
- Test it on fake languages
- Test it on a mix of languages
