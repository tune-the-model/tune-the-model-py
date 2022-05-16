import os

import pytest
import pandas as pd
from datasets import load_dataset

import model_one


@pytest.fixture(scope='session')
def configured_model_one():
    model_one.cli.API_KEY = os.environ.get("MODEL_ONE_KEY")


@pytest.fixture
def classifier(configured_model_one, tmp_path):
    dataset = load_dataset("tweet_eval", "irony")
    train = pd.DataFrame(dataset['train'])
    validation = pd.DataFrame(dataset['validation'])

    model = model_one.train_classifier(
        tmp_path / 'model-one-tweet_eval-irony.json',
        train['text'], 
        train['label'], 
        validation['text'], 
        validation['label'],
        train_iters=1
    )

    yield model

    model.delete()


def test_train_classifier(classifier):
    assert classifier.status in {
        model_one.ModelOneStatus.READY, 
        model_one.ModelOneStatus.TRAINING, 
        model_one.ModelOneStatus.TRAIN_REQUESTED,
        model_one.ModelOneStatus.INQUEUE
    }