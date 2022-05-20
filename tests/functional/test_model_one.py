import os

import pytest
import pandas as pd
from datasets import load_dataset

import model_one


@pytest.fixture(scope='session')
def configured_model_one():
    model_one.set_api_key(os.environ.get("MODEL_ONE_KEY"))


@pytest.fixture(scope="session")
def dataset():
    yield load_dataset("tweet_eval", "irony")


@pytest.fixture(scope="module")
def classifier(configured_model_one, tmpdir_factory, dataset):
    train = pd.DataFrame(dataset['train'])
    validation = pd.DataFrame(dataset['validation'])

    model = model_one.train_classifier(
        tmpdir_factory.mktemp("models").join("model-one-tweet_eval-irony.json"),
        train['text'], 
        train['label'], 
        validation['text'], 
        validation['label'],
        train_iters=10
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


@pytest.fixture(scope="module")
def trained_classifier(classifier):
    classifier.wait_for_training_finish()

    yield classifier


def test_trained_classifier(trained_classifier, dataset):
    assert trained_classifier.status == model_one.ModelOneStatus.READY

    validation = pd.DataFrame(dataset['validation'])

    res_validation = []
    for text in dataset['validation']['text']:
        res_validation += trained_classifier.classify(input=text)

    assert len(res_validation) > 0