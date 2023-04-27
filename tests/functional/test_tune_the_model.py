from datasets import load_dataset
import pandas as pd
import pytest

import tune_the_model as ttm


TRAIN_ITERS = 101


@pytest.fixture(scope="session")
def dataset():
    yield load_dataset("tweet_eval", "irony")

    
def test_fewshot_generate(configured_tune_the_model):
    output = ttm.generate("Tell me a joke")

    assert len(output) > 0
