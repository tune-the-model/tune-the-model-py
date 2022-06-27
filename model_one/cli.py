import os
import json

from enum import Enum
from functools import wraps
from time import sleep
from typing import List, Union

from pandas import Series
from numpy import ndarray
from urllib3 import Retry

from model_one.resource import (
    ModelOneAPI,
    ModelOneException,
)


MINIMUM_ENTRIES = 32


class ModelOneStatus(str, Enum):
    READY = "ready"
    CREATED = "created"
    DATASETS_LOADED = "datasetsloaded"
    TRAIN_REQUESTED = "trainrequested"
    TRAINING = "training"
    FAILED = "fitfailed"
    READY_TO_FIT = "readytofit"
    INQUEUE = "inqueue"


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

        data = json.dumps(data, default=_default)

        def MB(i):
            return i / 1024 ** 2

        upper_limit = 1.5 * 1024 ** 2
        if len(data) > upper_limit:
            raise ModelOneException(
                f"Payload exceeds the limit {MB(upper_limit):0.1f}MB with size of {MB(len(data)):0.2f}MB"
            )

        return ModelOneAPI.upload_file(self._id, data=data)

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

        if all(data is not None for data in [X, y]):
            from sklearn.model_selection import train_test_split
            train_X, validate_X, train_y, validate_y = train_test_split(
                X, y, train_size=train_size, test_size=test_size,
                random_state=random_state, shuffle=shuffle
            )

        train_file = ModelOneFile.create("train", self.type)
        train_file.upload(train_X, train_y)

        validate_file = ModelOneFile.create("val", self.type)
        validate_file.upload(validate_X, validate_y)

        train_file.wait_for_uploading_finish()
        validate_file.wait_for_uploading_finish()

        self.bind(train_file, validate_file)

        if self.status is not ModelOneStatus.DATASETS_LOADED:
            raise ModelOneException("Dataset is required")

        ModelOneAPI.fit(self._id)
        self._update_status()

        return self

    @inited
    def generate(self, input: str):
        """Generates a suffix based on an input prefix.

        Args:
            input : Prefix for generating a suffix.

        Returns:
            Generated text.

        Raises:
            ModelOneException: If anything bad happens.
        """
        r = ModelOneAPI.generate(self._id, input)
        return r["answer"]["responses"][0]["response"]

    @inited
    def classify(self, input: str):
        """Predicts a probability distribution over a set of classes given an input.

        Args:
            input : String to classify.

        Returns:
            A probability distribution over a set of classes.

        Raises:
            ModelOneException: If anything bad happens.
        """
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
    """Train the generator according to the given training data.

    Examples:
        The following snippet shows how to train a generator using the splitted train and validation data sets.

    .. code-block:: python

        import model_one


        train_inputs = ["алый", "альбом"] * 32
        train_outputs = ["escarlata", "el álbum"] * 32
        validation_inputs = ["бассейн", "бахрома"] * 32
        validation_outputs = ["libre", "flecos"] * 32

        model = model_one.train_generator(
            "classifier.json",
            train_inputs,
            train_outputs,
            validation_inputs,
            validation_outputs,
        )

    Args:
        filename : The path to a local file used to save the model info.
        train_X : Training data.
        train_y : Target class labels.
        validate_X : Validation data set used for controlling the training quality.
          The poor quality of this data set may lead to over-fitting.
        validate_y : Validation class labels.
        train_iters : Controls the number of train iterations.
        X : Training and validation data sets in one. It will be spltted with
          the help of sklearn.model_selection.train_test_split for you before
          uploading.
        y : Class labels.
        test_size : If float, should be between 0.0 and 1.0 and represent the
          proportion of the dataset to include in the validation split. If int,
          represents the absolute number of validation samples. If None, the value is
          set to the complement of the train size. If train_size is also None, it
          will be set to 0.25.
        train_size : If float, should be between 0.0 and 1.0 and represent the
          proportion of the dataset to include in the train split. If int,
          represents the absolute number of train samples. If None, the value is
          automatically set to the complement of the test size.
        shuffle : Whether or not to shuffle the data before splitting.
        random_state : Controls the shuffling applied to the data before
          applying the split. Pass an int for reproducible output across multiple
          function calls.

    Returns:
        The model object.

    Raises:
        ModelOneException: If anything bad happens.
    """
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
    """Train the classifier according to the given training data.

    Examples:
        The following snippet shows how to train a classifier using the splitted train and validation data sets.

    .. code-block:: python

        from datasets import load_dataset

        import model_one


        dataset = load_dataset("tweet_eval", "irony")

        train = pd.DataFrame(dataset["train"])
        validation = pd.DataFrame(dataset["validation"])

        model = model_one.train_classifier(
            "classifier.json",
            train["text"],
            train["label"],
            validation["text"],
            validation["label"],
        )

    Args:
        filename : The path to a local file used to save the model info.
        train_X : Training data.
        train_y : Target class labels.
        validate_X : Validation data set used for controlling the training quality.
          The poor quality of this data set may lead to over-fitting.
        validate_y : Validation class labels.
        train_iters : Controls the number of train iterations.
        num_classes : Creates a model for a multiclass classification task with <number of classes> classes.
        X : Training and validation data sets in one. It will be spltted with
          the help of sklearn.model_selection.train_test_split for you before
          uploading.
        y : Class labels.
        test_size : If float, should be between 0.0 and 1.0 and represent the
          proportion of the dataset to include in the validation split. If int,
          represents the absolute number of validation samples. If None, the value is
          set to the complement of the train size. If train_size is also None, it
          will be set to 0.25.
        train_size : If float, should be between 0.0 and 1.0 and represent the
          proportion of the dataset to include in the train split. If int,
          represents the absolute number of train samples. If None, the value is
          automatically set to the complement of the test size.
        shuffle : Whether or not to shuffle the data before splitting.
        random_state : Controls the shuffling applied to the data before
          applying the split. Pass an int for reproducible output across multiple
          function calls.

    Returns:
        The model object.

    Raises:
        ModelOneException: If anything bad happens.
    """
    model = ModelOne.create_classifier(filename, train_iters, num_classes)
    model.fit(train_X, train_y, validate_X, validate_y, X, y,
              test_size, train_size, shuffle, random_state)
    return model
