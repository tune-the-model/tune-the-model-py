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


from pkg_resources import parse_version, get_distribution
import requests
import json
import logging


def get_latest_version():
    response = requests.get("https://pypi.python.org/pypi/tune-the-model/json").text
    return json.loads(response)['info']['version']


def warn_if_outdated():
    log = logging.getLogger(__name__)
    current_version = get_distribution('tune-the-model').version
    latest_version = get_latest_version()
    if parse_version(current_version) < parse_version(latest_version):
        log.warning('The package tune-the-model is out of date. Your version is %s, the latest is %s.'
                    % (current_version, latest_version))


warn_if_outdated()


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
