from beyondml.model_one import (
    create_classification,
    create_generative,
    get_model,
    models,
    BeyondmlModel
)

import os

api_key = os.environ.get("BEYONDML_API_KEY")
api_url = 'https://api.beyond.ml'

__all__ = [
    'api',
    'api_key',
    'api_url',
    'create_classification',
    'create_generative',
    'get_model',
    'models',
    'BeyondmlModel'
]