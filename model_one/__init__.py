from model_one.model_one import (
    create_classification,
    create_generative,
    get_model,
    models,
    BeyondmlModel
)

import os

API_KEY = os.environ.get("BEYONDML_API_KEY")
api_url = 'https://api.beyond.ml'

__all__ = [
    'api',
    'API_KEY',
    'api_url',
    'create_classification',
    'create_generative',
    'get_model',
    'models',
    'BeyondmlModel'
]