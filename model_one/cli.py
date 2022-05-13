import os
import json
import requests

from enum import Enum
from functools import wraps
from time import sleep
from typing import Optional, List, Union

from pandas import Series
from numpy import ndarray


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
        "bind": ("POST", f"{API_URL}/v0/models/{{}}/bind"),
        "delete_model": ("DELETE", f"{API_URL}/v0/models/{{}}/delete"),
        "create_file": ("POST", f"{API_URL}/v0/files"),
        "files": ("GET", f"{API_URL}/v0/files"),
        "upload_file": ("POST", f"{API_URL}/v0/files/{{}}/upload"),
        "file_status": ("GET", f"{API_URL}/v0/files/{{}}/status"),
        "delete_file": ("DELETE", f"{API_URL}/v0/files/{{}}/delete"),
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
    def bind(cls, id: str, data: dict) -> dict:
        method, url = cls.V0["bind"]

        return cls._request(method, url.format(id), data=data)

    @classmethod
    def status(cls, id: str) -> dict:
        method, url = cls.V0["status"]

        return cls._request(method, url.format(id))

    @classmethod
    def fit(cls, id: str) -> dict:
        method, url = cls.V0["fit"]

        return cls._request(method, url.format(id))

    @classmethod
    def create_file(cls, data: dict) -> dict:
        return cls._request(*cls.V0["create_file"], data=json.dumps(data))

    @classmethod
    def upload_file(cls, id: str, data: dict) -> dict:
        method, url = cls.V0["upload_file"]

        return cls._request(method, url.format(id), data=data)

    @classmethod
    def delete_model(cls) -> dict:
        return cls._request(*cls.V0["delete_model"])

    @classmethod
    def delete_file(cls) -> dict:
        return cls._request(*cls.V0["delete_file"])

    @classmethod
    def file_status(cls, id: str) -> dict:
        method, url = cls.V0["file_status"]

        return cls._request(method, url.format(id))

    @classmethod
    def files(cls) -> dict:
        return cls._request(*cls.V0["files"])


class ModelOneStatus(str, Enum):
    READY = "ready"
    CREATED = "created"
    DATASETS_LOADED = "datasetsloaded"
    TRAIN_REQUESTED = "trainrequested"
    TRAINING = "readytofit"
    FAILED = "fitfailed"


class ModelOneFileStatus(str, Enum):
    READY = "fileready"
    CREATED = "filecreated"
    INQUEUE = "fileinqueue"
    LOADING = "fileloading"
    LOADED = "fileloaded"
    FAILED = "filefailed"


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


class ModelOneFile():
    _id: str = None
    _status: str = None
    _task_type: str = None
    _file_name: str = None

    def __init__(self, file_id: str, status: str, task_type: str, file_name: str, *args, **kwargs):
        self._id = file_id
        self._status = status
        self._task_type = task_type
        self._file_name = file_name

    @classmethod
    def from_dict(cls, model: dict) -> 'ModelOneFile':
        return cls(**model)

    @classmethod
    def from_id(cls, id: str) -> 'ModelOneFile':
        r = ModelOneAPI.file_status(id)
        return cls.from_dict(r)

    @classmethod
    def create(cls, file_name: str, task_type: ModelOneType) -> 'ModelOneFile':
        r = ModelOneAPI.create_file(
            {"name": file_name, "task_type": task_type.value})
        return cls.from_dict(r)

    @classmethod
    def load(cls, filename: str) -> 'ModelOneFile':
        with open(filename, "r") as fl:
            data = json.load(fl)
            return cls.from_dict(data)

    @classmethod
    def load_or_create(cls, filename: str, file_name: str, task_type: ModelOneType) -> 'ModelOneFile':
        if os.path.isfile(filename):
            model = cls.load(filename)
            if model.type != task_type:
                raise ModelOneException(
                    f"Model in the file is not {task_type}")
        else:
            model = cls.create(file_name, task_type)
            model.save(filename)
        return model

    @inited
    def save(self, filename):
        with open(filename, "w") as fl:
            json.dump({
                "id": self._id,
                "name": self._file_name,
                "status": self._status,
                "task_type": self._task_type,
            }, fl)

    @property
    def is_inited(self) -> bool:
        return self._id is not None

    @property
    @inited
    def is_ready(self):
        return self.status is ModelOneFileStatus.READY

    @property
    @inited
    def status(self) -> ModelOneFileStatus:
        status = self._update_status()
        return ModelOneFileStatus(status.lower())

    @property
    @inited
    def type(self) -> ModelOneType:
        return ModelOneType(self._task_type.lower())

    @property
    @inited
    def id(self) -> str:
        return self._id

    @inited
    def _update_status(self) -> str:
        r = ModelOneAPI.file_status(self._id)
        self._status = status = r["status"]
        return status

    @inited
    def delete(self) -> str:
        r = ModelOneAPI.delete_file(self._id)
        return r

    @inited
    def wait_for_uploading_finish(self, sleep_for: int = 1):
        while self.status is not ModelOneFileStatus.READY:
            sleep(sleep_for)

    @inited
    def upload(
        self,
        X: Union[list, Series, ndarray],
        y: Union[list, Series, ndarray]
    ) -> dict:
        if any(len(data) < MINIMUM_ENTRIES for data in [
            X, y
        ]):
            raise ModelOneException(
                f"Dataset must contain at least {MINIMUM_ENTRIES} elements")

        data = {
            "inputs": X,
        }

        key = {
            ModelOneType.GENERATOR: "outputs",
            ModelOneType.CLASSIFIER: "classes",
        }[self.type]

        data[key] = y

        def _default(val):
            if isinstance(val, Series) or isinstance(val, ndarray):
                return val.tolist()

            raise ModelOne(
                f"Value of type '{type(val)}' can not be serialized")

        return ModelOneAPI.upload_file(self._id, data=json.dumps(data, default=_default))

    @classmethod
    def files(cls) -> List['ModelOne']:
        r = ModelOneAPI.files()

        return [
            cls.from_dict(data) for data in r.get("files", [])
        ]

    def __repr__(self) -> str:
        return str(
            {
                "id": self._id,
                "status": self._status,
                "task_type": self._task_type,
                "name": self._file_name,
            }
        )

    def __str__(self) -> str:
        return str(
            {
                "id": self._id,
                "status": self._status,
                "task_type": self._task_type,
                "name": self._file_name,
            }
        )


class ModelOne():
    _id: str = None
    _status: str = None
    _model_type: str = None
    _model_user_name: str = None

    def __init__(self, model_name: str, status: str, model_type: str, *args, **kwargs):
        self._id = model_name
        self._status = status
        self._model_type = model_type
        self._model_user_name = kwargs["user_name"] if "user_name" in kwargs else None

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
    def create_classifier(cls, filename: str, train_iters: int = None, num_classes: int = None):
        model = {"model_type": "classifier"}

        if train_iters:
            model["model_params"] = {"train_iters": train_iters}
        if num_classes:
            model["model_params"] = {"num_classes": num_classes}

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
                raise ModelOneException(
                    f"Model in the file is not {data['model_type']}")
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
    def delete(self) -> str:
        r = ModelOneAPI.delete_model(self._id)
        return r

    @inited
    def fit(
        self,
        train_X: Union[list, Series, ndarray, None] = None,
        train_y: Union[list, Series, ndarray, None] = None,
        validate_X: Union[list, Series, ndarray, None] = None,
        validate_y: Union[list, Series, ndarray, None] = None,
        X: Union[list, Series, ndarray, None] = None,
        y: Union[list, Series, ndarray, None] = None,
        test_size=None,
        train_size=None,
        shuffle=True,
        random_state=None
    ) -> 'ModelOne':
        if self.status in {ModelOneStatus.READY, ModelOneStatus.TRAINING, ModelOneStatus.TRAIN_REQUESTED}:
            return self

        if all(data is not None for data in [train_X, train_y, validate_X, validate_y]):
            self.upload(train_X, train_y, validate_X, validate_y)
        elif all(data is not None for data in [X, y]):
            from sklearn.model_selection import train_test_split
            train_X, validate_X, train_y, validate_y = train_test_split(
                X, y, train_size=train_size, test_size=test_size, random_state=random_state, shuffle=shuffle)
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
    def bind(self, train_file: ModelOneFile, validate_file: ModelOneFile) -> 'ModelOne':
        ModelOneAPI.bind(self._id, data={
                         "train_file": train_file.id, "validate_file": validate_file.id})
        self._update_status()
        return self

    @inited
    def upload(
        self,
        train_X: Union[list, Series, ndarray],
        train_y: Union[list, Series, ndarray],
        validate_X: Union[list, Series, ndarray],
        validate_y: Union[list, Series, ndarray]
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
            if isinstance(val, Series) or isinstance(val, ndarray):
                return val.tolist()

            raise ModelOne(
                f"Value of type '{type(val)}' can not be serialized")

        return ModelOneAPI.upload(self._id, data=json.dumps(data, default=_default))

    def __repr__(self) -> str:
        return str(
            {
                "id": self._id,
                "status": self._status,
                "model_type": self._model_type,
                "user_name": self._model_user_name,
            }
        )

    def __str__(self) -> str:
        return str(
            {
                "id": self._id,
                "status": self._status,
                "model_type": self._model_type,
                "user_name": self._model_user_name,
            }
        )


def train_generator(
    filename: str,
    train_X: Union[list, Series, ndarray, None] = None,
    train_y: Union[list, Series, ndarray, None] = None,
    validate_X: Union[list, Series, ndarray, None] = None,
    validate_y: Union[list, Series, ndarray, None] = None,
    train_iters: int = None,
    X: Union[list, Series, ndarray, None] = None,
    y: Union[list, Series, ndarray, None] = None,
    test_size=None,
    train_size=None,
    shuffle=True,
    random_state=None
) -> ModelOne:
    model = ModelOne.create_generator(filename, train_iters)
    model.fit(train_X, train_y, validate_X, validate_y, X, y,
              test_size, train_size, shuffle, random_state)
    return model


def train_classifier(
    filename: str,
    train_X: Union[list, Series, ndarray, None] = None,
    train_y: Union[list, Series, ndarray, None] = None,
    validate_X: Union[list, Series, ndarray, None] = None,
    validate_y: Union[list, Series, ndarray, None] = None,
    train_iters: int = None,
    num_classes: int = None,
    X: Union[list, Series, ndarray, None] = None,
    y: Union[list, Series, ndarray, None] = None,
    test_size=None,
    train_size=None,
    shuffle=True,
    random_state=None
) -> ModelOne:
    model = ModelOne.create_classifier(filename, train_iters, num_classes)
    model.fit(train_X, train_y, validate_X, validate_y, X, y,
              test_size, train_size, shuffle, random_state)
    return model
