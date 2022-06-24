import os

from datasets import load_dataset
import pandas as pd
import pytest
import requests

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
        tmpdir_factory.mktemp("models").join("classifier.json"),
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


def test_train_classifier_with_large_dataset(configured_model_one, tmpdir_factory, dataset):
    data = pd.DataFrame(dataset["train"])

    data = pd.concat([data] * 10)

    model = model_one.ModelOne.create_classifier(
        tmpdir_factory.mktemp("models").join("classifier-with-large-dataset.json"),
        train_iters=10,
    )

    with pytest.raises(model_one.ModelOneException) as exc:
        model.fit(
            X=data["text"],
            y=data["label"],
        )
    assert "Payload exceeds the limit" in str(exc.value)

    model.delete()


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


@pytest.fixture(scope="module")
def generator(configured_model_one, tmpdir_factory, dataset):
    train_inputs = ["алый", "альбом"] * 32
    train_outputs = ["escarlata", "el álbum"] * 32
    validation_inputs = ["бассейн", "бахрома"] * 32
    validation_outputs = ["libre", "flecos"] * 32

    model = model_one.train_generator(
        tmpdir_factory.mktemp("models").join("generator.json"),
        train_inputs,
        train_outputs,
        validation_inputs,
        validation_outputs,
        train_iters=10
    )

    yield model

    model.delete()


def test_generator(generator):
    assert generator.status in {
        model_one.ModelOneStatus.READY,
        model_one.ModelOneStatus.TRAINING,
        model_one.ModelOneStatus.TRAIN_REQUESTED,
        model_one.ModelOneStatus.INQUEUE
    }


@pytest.fixture(scope="module")
def trained_generator(generator):
    generator.wait_for_training_finish()

    yield generator


def test_trained_generator(trained_generator):
    assert trained_generator.status == model_one.ModelOneStatus.READY

    assert len(trained_generator.generate("бассейн")) > 0