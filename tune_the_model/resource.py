from typing import Optional, List, Tuple
import json
import os

import requests


API_KEY = os.environ.get("TTM_API_KEY")
API_URL = "https://api.tunethemodel.com"


def set_api_key(api_key):
    global API_KEY
    API_KEY = api_key


class TuneTheModelException(RuntimeError):
    pass


class TuneTheModelAPI():
    API_URL: Optional[str] = None

    V0: dict = {
        "create": ("POST", "{}/v0/models"),
        "models": ("GET", "{}/v0/models"),
        "vanilla_generate": ("POST", "{}/v0/generate"),
        "generate": ("GET", "{}/v0/models/{}/generate"),
        "classify": ("GET", "{}/v0/models/{}/classify"),
        "upload": ("POST", "{}/v0/models/{}/upload"),
        "status": ("GET", "{}/v0/models/{}/status"),
        "fit": ("POST", "{}/v0/models/{}/fit"),
        "bind": ("POST", "{}/v0/models/{}/bind"),
        "delete_model": ("DELETE", "{}/v0/models/{}/delete"),
        "create_file": ("POST", "{}/v0/files"),
        "files": ("GET", "{}/v0/files"),
        "upload_file": ("POST", "{}/v0/files/{}/upload"),
        "file_status": ("GET", "{}/v0/files/{}/status"),
        "delete_file": ("DELETE", "{}/v0/files/{}/delete"),
    }

    @classmethod
    def _get_V0(cls, method: str, *args, **kwargs) -> Tuple[str, str]:
        method, url = cls.V0[method]

        return method, url.format(cls.API_URL or API_URL, *args, **kwargs)

    @classmethod
    def _request(cls, method: str, url: str, params: Optional[dict] = None, data: Optional[dict] = None) -> dict:
        headers = {"Authorization": API_KEY}

        if data is not None:
            headers["Content-Type"] = "application/json"

        retry_strategy = requests.packages.urllib3.util.retry.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "PUT", "DELETE", "POST", "OPTIONS", "TRACE"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        http = requests.Session()
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        try:
            r = http.request(
                method,
                url,
                params=params,
                data=data,
                headers=headers
            )
        except requests.exceptions.RetryError as e:
            raise TuneTheModelException(e)

        if r.status_code != 200:
            raise TuneTheModelException(r.text)

        return r.json()

    @classmethod
    def create(cls, data: dict) -> dict:
        return cls._request(*cls._get_V0("create"), data=json.dumps(data))

    @classmethod
    def models(cls) -> dict:
        return cls._request(*cls._get_V0("models"))

    @classmethod
    def classify(cls, id: str, input: str) -> dict:
        return cls._request(*cls._get_V0("classify", id), params={"input": input})

    @classmethod
    def vanilla_generate(cls, data: dict) -> dict:
        return cls._request(*cls._get_V0("vanilla_generate"), data=json.dumps(data))

    @classmethod
    def generate(cls, id: str, input: str) -> dict:
        return cls._request(*cls._get_V0("generate", id), params={"input": input})

    @classmethod
    def bind(cls, id: str, data: dict) -> dict:
        return cls._request(*cls._get_V0("bind", id), data=json.dumps(data))

    @classmethod
    def status(cls, id: str) -> dict:
        return cls._request(*cls._get_V0("status", id))

    @classmethod
    def fit(cls, id: str) -> dict:
        return cls._request(*cls._get_V0("fit", id))

    @classmethod
    def create_file(cls, data: dict) -> dict:
        return cls._request(*cls._get_V0("create_file"), data=json.dumps(data))

    @classmethod
    def upload_file(cls, id: str, data: dict) -> dict:
        return cls._request(*cls._get_V0("upload_file", id), data=data)

    @classmethod
    def delete_model(cls, id: str) -> dict:
        return cls._request(*cls._get_V0("delete_model", id))

    @classmethod
    def delete_file(cls, id: str) -> dict:
        return cls._request(*cls._get_V0("delete_file", id))

    @classmethod
    def file_status(cls, id: str) -> dict:
        return cls._request(*cls._get_V0("file_status", id))

    @classmethod
    def files(cls) -> dict:
        return cls._request(*cls._get_V0("files"))