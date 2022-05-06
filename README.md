# Model One python wrapper

[Model One](https://beyond.ml/model-one) is a few-shot AutoML system.

Model One can do almost anything that requires understanding or generating natural language. Model One is able to solve tasks in 12 languages: English, Spanish, Portuguese, Russian, Turkish, French, German, Italian, Arabic, Polish, Dutch, and Hebrew.

This package provides a simple wrapper for using our api.

Using `model-one` package allows you to train and apply models.

## Get started

Firstly fill out the [form](https://beyond.ml/model-one#rec435480002) to get a key to access the API. We will send you the key within a day.

### Install
To install the package just use `pip install model-one`.

### Usage

```py
import model_one
import pandas as pd

model_one.set_api_key('YOUR_API_KEY')

# load datasets
tdf = pd.read_csv('train.csv')
vdf = pd.read_csv('test.csv')

# Call one method. It will do everything for you:
# create a model, save it to the file, upload datasets and put the model in the queue for training.
model = model_one.train_generator(
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
