# Tune The Model python wrapper

[Tune The Model](https://tunethemodel.com) is a few-shot AutoML system.

Tune The Model can do almost anything that requires understanding or generating natural language. It is able to solve tasks in 12 languages: English, Spanish, Portuguese, Russian, Turkish, French, German, Italian, Arabic, Polish, Dutch, and Hebrew.

This package provides a simple wrapper for using our api.

Using `tune-the-model` package allows you to train and apply models.

## Documentation

You can find the documentation at our [Tune The Model API docs site](https://tune-the-model.github.io/tune-the-model-docs/index.html).

## Just try

We have fine-tuned several models. You can use the [notebook](https://colab.research.google.com/github/beyondml/model-one-py/blob/main/playbook.ipynb) to try them out. You can [get the token](https://tunethemodel.com) to fine tune your own model.

## Getting started

Firstly fill out the [form](https://tunethemodel.com) to get a key to access the API. We will send you the key within a day.

### Installation

To install the package just use `pip install -U tune-the-model`.

### Usage

```py
import tune_the_model as ttm
import pandas as pd

ttm.set_api_key('YOUR_API_KEY')

# load datasets
tdf = pd.read_csv('train.csv')
vdf = pd.read_csv('test.csv')

# Call one method. It will do everything for you:
# create a model, save it to the file, upload datasets and put the model in the queue for training.
model = ttm.tune_generator(
    'filename.json',
    tdf['inputs'], tdf['outputs'],
    vdf['inputs'], vdf['outputs']
)

# wait...
# a few hours
# while our GPUs train your model
model.wait_for_training_finish()

print(model.status)
print(model.is_ready)

# inference!
the_answer = model.generate('The Answer to the Ultimate Question of Life, the Universe, and Everything')
print(the_answer)
```
