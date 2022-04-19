from array import array
import json
from time import time
import requests

import beyondml


def _models_api():
    return '{}/v0/models'.format(beyondml.api_url)


def _get_auth_header():
    return {'Authorization': beyondml.api_key}


class BeyondmlModel():
    def __init__(self, model=None):
        if model is not None:
            self._id = model['model_name']
            self._status = model['status']
            self._model_type = model['model_type']
        return

    def __repr__(self):
        return str({'id': self._id, 'status': self._status, 'model_type': self._model_type})

    def __str__(self):
        return str({'id': self._id, 'status': self._model_type, 'model_type': self._model_type})

    def _api_post_request(self, handler, data):
        if not self._id:
            raise Exception('Initialize the model.')
        headers = _get_auth_header()
        headers['Content-Type'] = 'application/json'
        url = '{}/{}/{}'.format(_models_api(), self._id, handler)
        r = requests.post(url, headers=headers, data=data)
        if r.status_code != 200:
            raise Exception(r.text)
        self._status = r.json()['status']
        return r.json()

    def fit(self):
        return self._api_post_request('fit', '')

    def is_ready(self):
        return self._status.lower() == 'ready'

    def upload_arrays(self, train_X, train_y, validate_X, validate_y):
        def build_data(self):
            data = {
                'train_dataset': {
                    'inputs': train_X,
                },
                'validate_dataset': {
                    'inputs': validate_X,
                }
            }
            if self._model_type == 'generative':
                data['train_dataset']['outputs'] = train_y
                data['validate_dataset']['outputs'] = validate_y
            elif self._model_type == 'classification':
                data['train_dataset']['classes'] = train_y
                data['validate_dataset']['classes'] = validate_y
            else:
                raise Exception('Unknown model type.')
            return json.dumps(data)

        return self._api_post_request('upload', build_data(self))

    def upload(self, train_X, train_y, validate_X, validate_y):
        from pandas import Series
        if not (isinstance(train_X, Series) and isinstance(train_X, Series)
                and isinstance(train_X, Series) and isinstance(train_X, Series)):
            raise Exception('Try upload_arrays.')

        return self.upload_arrays(train_X.tolist(), train_y.tolist(), validate_X.tolist(), validate_y.tolist())

    def _inference(self, method, input):
        if not input:
            raise Exception('prompt must not be empty')

        r = requests.get('{}/{}/{}?input={}'.format(_models_api(),
                         self._id, method, input), headers=_get_auth_header())
        if r.status_code != 200:
            raise Exception(r.text)
        return r.json()

    def generate(self, input):
        return self._inference('generate', input)

    def classify(self, input):
        return self._inference('classify', input)

    def status(self):
        if not self._id:
            raise Exception('Initialize the model.')
        r = requests.get('{}/{}/{}'.format(_models_api(),
                         self._id, 'status'), headers=_get_auth_header())
        if r.status_code != 200:
            raise Exception(r.text)
        self._status = r.json()['status']
        return r.json()

    def save(self, filename):
        with open(filename, 'w') as fl:
            json.dump({
                'model_name': self._id,
                'status': self._status,
                'model_type': self._model_type,
            }, fl)


def _create(data):
    headers = _get_auth_header()
    headers['Content-Type'] = 'application/json'
    r = requests.post(_models_api(), headers=headers, data=json.dumps(data))
    if r.status_code != 200:
        raise Exception(r.text)
    return BeyondmlModel(r.json())


def create_generative():
    o = {'model_type': 'generative'}
    return _create(o)


def create_classification():
    o = {'model_type': 'classification'}
    return _create(o)


def load(filename):
    with open(filename, 'r') as fl:
        return BeyondmlModel(json.load(fl))


def models():
    r = requests.get(_models_api(), headers=_get_auth_header())
    if r.status_code != 200:
        raise Exception(r.text)

    res = []
    for k in r.json()['models']:
        res.append(BeyondmlModel(k))
    return res


def get_model(id):
    r = requests.get('{}/{}/status'.format(_models_api(), id),
                     headers=_get_auth_header())
    if r.status_code != 200:
        raise Exception(r.text)
    return BeyondmlModel(r.json())
