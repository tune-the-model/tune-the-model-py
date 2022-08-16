from importlib.metadata import version
from pkg_resources import parse_version
import requests
import json


def get_latest_version():
    response = requests.get("https://pypi.python.org/pypi/tune-the-model/json").text
    return json.loads(response)['info']['version']


if (parse_version(version('tune-the-model')) < parse_version(get_latest_version())):
    raise Exception("Update package")


from tune_the_model.cli import (
    TuneTheModelStatus,
    TuneTheModelType,
    TuneTheModel,
    TuneTheModelFile,
    TuneTheModelFileStatus,
    tune_generator,
    tune_classifier,
    generate,
)
from tune_the_model.resource import (
    API_URL,
    TuneTheModelAPI,
    TuneTheModelException,
    set_api_key
)

__all__ = [
    "tune_the_model",
    "set_api_key",
    "API_URL",
    "TuneTheModelException",
    "TuneTheModelStatus",
    "TuneTheModelType",
    "TuneTheModel",
    "TuneTheModelAPI",
    "TuneTheModelFile",
    "TuneTheModelFileStatus",
    "tune_generator",
    "tune_classifier",
    "generate",
]
