import os

import pytest

import tune_the_model as ttm


@pytest.fixture(scope='session')
def configured_tune_the_model():
    ttm.set_api_key("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwidXNlcl9pZCI6Imlla2FyX3Bvdl95YW5kZXhfdGVhbXJ1IiwiaWF0IjoxNTE2MjM5MDIyLCJJU19SRUFET05MWSI6ZmFsc2V9.kMv2xJEBZZfSVL9VqQCmq6j46pRzMkF-DQcKLtFj-ns")
