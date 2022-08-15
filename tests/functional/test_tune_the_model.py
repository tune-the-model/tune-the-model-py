import os

from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import pytest
import multiprocessing as mp

import tune_the_model as ttm


TRAIN_ITERS = 101


@pytest.fixture(scope='session')
def configured_tune_the_model():
    ttm.set_api_key(os.environ.get("TTM_API_KEY"))


@pytest.fixture(scope="session")
def dataset():
    yield load_dataset("tweet_eval", "irony")


@pytest.fixture(scope="module")
def classifier(configured_tune_the_model, tmpdir_factory, dataset):
    train = pd.DataFrame(dataset['train'])
    validation = pd.DataFrame(dataset['validation'])

    model = ttm.tune_classifier(
        tmpdir_factory.mktemp("models").join("classifier.json"),
        train['text'],
        train['label'],
        validation['text'],
        validation['label'],
        train_iters=TRAIN_ITERS
    )

    yield model

    model.delete()


def test_train_classifier(classifier):
    assert classifier.status in {
        ttm.TuneTheModelStatus.READY,
        ttm.TuneTheModelStatus.TRAINING,
        ttm.TuneTheModelStatus.TRAIN_REQUESTED,
        ttm.TuneTheModelStatus.INQUEUE
    }


def test_fewshot_generate(configured_tune_the_model):
    output = ttm.generate("Tell me a joke")

    assert len(output) > 0


def test_train_classifier_with_large_dataset(configured_tune_the_model, tmpdir_factory, dataset):
    data = pd.DataFrame(dataset["train"])

    data = pd.concat([data] * 50)

    model = ttm.TuneTheModel.create_classifier(
        tmpdir_factory.mktemp("models").join("classifier-with-large-dataset.json"),
        train_iters=TRAIN_ITERS,
    )

    with pytest.raises(ttm.TuneTheModelException) as exc:
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


def classify_many(model, data):
    with mp.Pool(5) as p:
        predictions = p.map(model.classify, data)
    return predictions


def test_trained_classifier(trained_classifier, dataset):
    assert trained_classifier.status == ttm.TuneTheModelStatus.READY

    res_validation = []
    for text in dataset['validation']['text']:
        res_validation += trained_classifier.classify(input=text)

    assert len(res_validation) > 0

    predictions = classify_many(trained_classifier, dataset['test']['text'])
    predictions = [int(prob[0] > 0.5) for prob in predictions]
    test_f1 = f1_score(dataset['test']['label'], predictions)
    assert test_f1 > 0.57


def test_trained_multiclass(configured_tune_the_model, tmpdir_factory):
    dataset = load_dataset("qanastek/MASSIVE", "ru-RU")

    train = pd.DataFrame(dataset['train'])
    validation = pd.DataFrame(dataset['validation'])
    test = pd.DataFrame(dataset['test'])

    model = ttm.tune_classifier(
        tmpdir_factory.mktemp("models").join("classifier.json"),
        train['utt'],
        train['intent'],
        validation['utt'],
        validation['intent'],
        num_classes=60,
        train_iters=TRAIN_ITERS
    )

    model.wait_for_training_finish()
    model.classify(input="поставь будильник на 9 утра")

    predictions = classify_many(model, test['utt'])
    predictions = np.argmax(predictions, axis=1)
    test_f1 = f1_score(test['intent'], predictions, average='macro')
    assert test_f1 > 0.70

    model.delete()


@pytest.fixture(scope="module")
def generator(configured_tune_the_model, tmpdir_factory, dataset):
    train_inputs = ["алый", "альбом"] * 32
    train_outputs = ["escarlata", "el álbum"] * 32
    validation_inputs = ["бассейн", "бахрома"] * 32
    validation_outputs = ["libre", "flecos"] * 32

    model = ttm.tune_generator(
        tmpdir_factory.mktemp("models").join("generator.json"),
        train_inputs,
        train_outputs,
        validation_inputs,
        validation_outputs,
        train_iters=TRAIN_ITERS
    )

    yield model

    model.delete()


def test_generator(generator):
    assert generator.status in {
        ttm.TuneTheModelStatus.READY,
        ttm.TuneTheModelStatus.TRAINING,
        ttm.TuneTheModelStatus.TRAIN_REQUESTED,
        ttm.TuneTheModelStatus.INQUEUE
    }


@pytest.fixture(scope="module")
def trained_generator(generator):
    generator.wait_for_training_finish()

    yield generator


def test_trained_generator(trained_generator):
    assert trained_generator.status == ttm.TuneTheModelStatus.READY

    assert len(trained_generator.generate("бассейн")) > 0
