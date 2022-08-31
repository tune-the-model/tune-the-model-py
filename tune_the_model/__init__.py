from pkg_resources import parse_version, get_distribution
import json
import logging

import requests

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


log = logging.getLogger(__name__)


def get_latest_version():
    response = requests.get("https://pypi.python.org/pypi/tune-the-model/json").text
    return json.loads(response)['info']['version']


def warn_if_outdated():
    current_version = get_distribution('tune-the-model').version
    latest_version = get_latest_version()
    if parse_version(current_version) < parse_version(latest_version):
        log.warning('Please update the package using `pip install -U tune-the-model`. Your version is %s, the latest is %s.'
                    % (current_version, latest_version))


try:
    warn_if_outdated()
except Exception:
    log.warn('Something went wrong during package version check. Please update the package regularly')


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
