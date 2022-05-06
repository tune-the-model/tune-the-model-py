import os
import json
import requests

from enum import Enum
from functools import wraps
from time import sleep
from typing import Optional, List, Union

from pandas import Series


API_KEY = os.environ.get("BEYONDML_API_KEY")
API_URL = "https://api.beyond.ml"
MINIMUM_ENTRIES = 32


def set_api_key(api_key):
    global API_KEY
    API_KEY = api_key


class ModelOneException(RuntimeError):
    pass


class ModelOneAPI():
    V0: dict = {
        "create": ("POST", f"{API_URL}/v0/models"),
        "models": ("GET", f"{API_URL}/v0/models"),
        "generate": ("GET", f"{API_URL}/v0/models/{{}}/generate"),
        "classify": ("GET", f"{API_URL}/v0/models/{{}}/classify"),
        "upload": ("POST", f"{API_URL}/v0/models/{{}}/upload"),
        "status": ("GET", f"{API_URL}/v0/models/{{}}/status"),
        "fit": ("POST", f"{API_URL}/v0/models/{{}}/fit"),
    }

    @classmethod
    def _request(cls, method: str, url: str, params: Optional[dict] = None, data: Optional[dict] = None) -> dict:
        headers = {"Authorization": API_KEY, }

        if data is not None:
            headers["Content-Type"] = "application/json"

        r = requests.request(method, url, params=params,
                             data=data, headers=headers)

        if r.status_code != 200:
            raise ModelOneException(r.text)

        return r.json()

    @classmethod
    def create(cls, data: dict) -> dict:
        return cls._request(*cls.V0["create"], data=json.dumps(data))

    @classmethod
    def models(cls) -> dict:
        return cls._request(*cls.V0["models"])

    @classmethod
    def classify(cls, id: str, input: str) -> dict:
        method, url = cls.V0["classify"]

        return cls._request(method, url.format(id), params={"input": input})

    @classmethod
    def generate(cls, id: str, input: str) -> dict:
        method, url = cls.V0["generate"]

        return cls._request(method, url.format(id), params={"input": input})

    @classmethod
    def upload(cls, id: str, data: dict) -> dict:
        method, url = cls.V0["upload"]

        return cls._request(method, url.format(id), data=data)

    @classmethod
    def status(cls, id: str) -> dict:
        method, url = cls.V0["status"]

        return cls._request(method, url.format(id))

    @classmethod
    def fit(cls, id: str) -> dict:
        method, url = cls.V0["fit"]

        return cls._request(method, url.format(id))


class ModelOneStatus(str, Enum):
    READY = "ready"
    CREATED = "created"
    DATASETS_LOADED = "datasetsloaded"
    TRAIN_REQUESTED = "trainrequested"
    TRAINING = "readytofit"
    FAILED = "fitfailed"


class ModelOneType(str, Enum):
    GENERATOR = "generator"
    CLASSIFIER = "classifier"


def inited(m: callable):
    @wraps(m)
    def _wrapper(self: 'ModelOne', *args, **kwargs):
        if not self.is_inited:
            raise ModelOneException("Initialize the model")

        return m(self, *args, **kwargs)

    return _wrapper


class ModelOne():
    _id: str = None
    _status: str = None
    _model_type: str = None

    def __init__(self, model_name: str, status: str, model_type: str, *args, **kwargs):
        self._id = model_name
        self._status = status
        self._model_type = model_type

    @classmethod
    def from_dict(cls, model: dict) -> 'ModelOne':
        return cls(**model)

    @classmethod
    def from_id(cls, id: str) -> 'ModelOne':
        r = ModelOneAPI.status(id)
        return cls.from_dict(r)

    @classmethod
    def create(cls, data: dict) -> 'ModelOne':
        r = ModelOneAPI.create(data)
        return cls.from_dict(r)

    @classmethod
    def create_classifier(cls, filename: str, train_iters: int = None):
        model = {"model_type": "classifier"}

        if train_iters:
            model["model_params"] = {"train_iters": train_iters}

        return cls.load_or_create(filename, model)

    @classmethod
    def create_generator(cls, filename: str, train_iters: int = None):
        model = {"model_type": "generator"}

        if train_iters:
            model["model_params"] = {"train_iters": train_iters}

        return cls.load_or_create(filename, model)

    @classmethod
    def models(cls) -> List['ModelOne']:
        r = ModelOneAPI.models()

        return [
            cls.from_dict(data) for data in r.get("models", [])
        ]

    @classmethod
    def load(cls, filename: str) -> 'ModelOne':
        with open(filename, "r") as fl:
            data = json.load(fl)
            return cls.from_dict(data)

    @classmethod
    def load_or_create(cls, filename: str, data: dict) -> 'ModelOne':
        if os.path.isfile(filename):
            model = cls.load(filename)
            if model.type != data["model_type"]:
                raise ModelOneException(f"Model in the file is not {data['model_type']}")
        else:
            model = cls.create(data)
            model.save(filename)
        return model

    @inited
    def save(self, filename):
        with open(filename, "w") as fl:
            json.dump({
                "model_name": self._id,
                "status": self._status,
                "model_type": self._model_type,
            }, fl)

    @property
    def is_inited(self) -> bool:
        return self._id is not None

    @property
    @inited
    def is_ready(self):
        return self.status is ModelOneStatus.READY

    @property
    @inited
    def status(self) -> ModelOneStatus:
        status = self._update_status()
        return ModelOneStatus(status.lower())

    @property
    @inited
    def type(self) -> ModelOneType:
        return ModelOneType(self._model_type.lower())

    @inited
    def _update_status(self) -> str:
        r = ModelOneAPI.status(self._id)
        self._status = status = r["status"]
        return status

    @inited
    def fit(
        self,
        train_X: Union[list, Series, None] = None,
        train_y: Union[list, Series, None] = None,
        validate_X: Union[list, Series, None] = None,
        validate_y: Union[list, Series, None] = None
    ) -> 'ModelOne':
        if self.status in {ModelOneStatus.READY, ModelOneStatus.TRAINING, ModelOneStatus.TRAIN_REQUESTED}:
            return self

        if all(data is not None for data in [train_X, train_y, validate_X, validate_y]):
            self.upload(train_X, train_y, validate_X, validate_y)

        if self.status is not ModelOneStatus.DATASETS_LOADED:
            raise ModelOneException("Dataset is required")

        ModelOneAPI.fit(self._id)
        self._update_status()

        return self

    @inited
    def generate(self, input: str):
        r = ModelOneAPI.generate(self._id, input)
        return r["answer"]["responses"][0]["response"]

    @inited
    def classify(self, input: str):
        r = ModelOneAPI.classify(self._id, input)
        return r["answer"]["scores"]

    @inited
    def wait_for_training_finish(self, sleep_for: int = 60):
        while self.status is not ModelOneStatus.READY:
            sleep(sleep_for)

    @inited
    def upload(
        self,
        train_X: Union[list, Series],
        train_y: Union[list, Series],
        validate_X: Union[list, Series],
        validate_y: Union[list, Series]
    ) -> dict:
        if any(len(data) < MINIMUM_ENTRIES for data in [
            train_X, train_y, validate_X, validate_y
        ]):
            raise ModelOneException(
                f"Dataset must contain at least {MINIMUM_ENTRIES} elements")

        data = {
            "train_dataset": {
                "inputs": train_X,
            },
            "validate_dataset": {
                "inputs": validate_X,
            }
        }

        key = {
            ModelOneType.GENERATOR: "outputs",
            ModelOneType.CLASSIFIER: "classes",
        }[self.type]

        data["train_dataset"][key] = train_y
        data["validate_dataset"][key] = validate_y

        def _default(val):
            if isinstance(val, Series):
                return val.tolist()

            raise ModelOne(
                f"Value of type '{type(val)}' can not be serialized")

        return ModelOneAPI.upload(self._id, data=json.dumps(data, default=_default))

    def __repr__(self) -> str:
        return str(
            {
                "id": self._id,
                "status": self._status,
                "model_type": self._model_type
            }
        )

    def __str__(self) -> str:
        return str(
            {
                "id": self._id,
                "status": self._model_type,
                "model_type": self._model_type
            }
        )


def train_generator(
    filename: str,
    train_X: Union[list, Series, None],
    train_y: Union[list, Series, None],
    validate_X: Union[list, Series, None],
    validate_y: Union[list, Series, None],
    train_iters: int = None
) -> ModelOne:
    model = ModelOne.create_generator(filename, train_iters)
    model.fit(train_X, train_y, validate_X, validate_y)
    return model


def train_classifier(
    filename: str,
    train_X: Union[list, Series, None],
    train_y: Union[list, Series, None],
    validate_X: Union[list, Series, None],
    validate_y: Union[list, Series, None],
    train_iters: int = None
) -> ModelOne:
    model = ModelOne.create_classifier(filename, train_iters)
    model.fit(train_X, train_y, validate_X, validate_y)
    return model
