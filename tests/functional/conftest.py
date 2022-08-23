import os

import pytest

import tune_the_model as ttm


@pytest.fixture(scope='session')
def configured_tune_the_model():
    ttm.set_api_key(os.environ.get("TTM_API_KEY"))
