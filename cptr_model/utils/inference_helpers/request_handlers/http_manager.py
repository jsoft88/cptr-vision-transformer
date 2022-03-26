import base64
import json
from typing import Any, Dict, List, Optional, Tuple
from PIL.Image import Image, open
from io import BytesIO
import requests
from cptr_model.utils.inference_helpers.inference_helper_mixin import InferenceHelperMixin
from cptr_model.utils.inference_helpers.request_handlers.base_auth_manager import BaseAuthManager


class HttpManager(InferenceHelperMixin):
    URL_REQ_FETCH = '/requests'
    REQ_PARAM_BATCH_SIZE = 'batchSize'
    URL_REQ_POST = '/prediction'

    def __init__(self, host: str, port: int, auth: Optional[BaseAuthManager], protocol: str = 'http') -> None:
        self.host = host
        self.port = port
        self.auth = auth
        self.protocol = protocol

    @staticmethod
    def post_request(url: str, body: Any) -> Any:
        response = requests.post(url, data=body)
        if response.status_code not in range(200, 300):
            raise Exception(f'POST Request to {url} failed. Code is {response.status_code} and description -> {response.content}')

        return json.loads(response.content)

    @staticmethod
    def get_request(url: str, params: Optional[Dict[str, str]] = None) -> Any:
        response = requests.get(url, params)
        if response.status_code not in range(200, 300):
            raise Exception(f'GET Request to {url} failed. Code is {response.status_code} and description -> {response.content}')

        #image will be base64 encoded
        json_body = json.loads(response.content)
        return json_body

    '''
    Response to have the format:
    {
        "items": [
            {
                "reqId": uuid1,
                "image": <base64_encoded_image>
            },
            {
                "reqId": uuid2,
                "image": <base64_encoded_image>
            },
            .
            .
            .,
            {
                "reqId": uuid(batch_size),
                "image": <base64_encoded_image>
            }
        ]
    }
    '''
    def get_input(self, batch_size: int) -> List[Tuple[str, Image]]:
        json_obj = HttpManager.get_request(f'{self.protocol}://{self.host}:{self.port}/{HttpManager.URL_REQ_FETCH}', {HttpManager.REQ_PARAM_BATCH_SIZE: batch_size})
        ret_val = []
        for obj in json_obj['items']:
            ret_val.append((obj['reqId'], open(BytesIO(base64.b64decode(obj['image'])))))
        
        return ret_val

    '''
    Body to post:
    {
        "predictions": [
            {
                "reqId": uuid1,
                "prediction": caption1
            },
            {
                "reqId": uuid2,
                "prediction": caption2
            },
            .
            .
            .,
            {
                "reqId": uuuid(batch_size),
                "prediction": caption_batch_size
            }
        ]
    }

    Response:
    {
        "success": true | false,
        "description": "description"
    }
    '''
    def post_prediction(self, predictions: List[Dict[Any, Any]]) -> Any:
        data_predictions = {}
        pred_objects = []
        for pred in predictions:
            pred_objects.append({'reqId': pred.keys()[0], 'prediction': pred.values()[0]})

        data_predictions['predictions'] = pred_objects
        post_data = json.dumps(data_predictions)
        resp = HttpManager.post_request(f'{self.protocol}://{self.host}:{self.port}/{HttpManager.URL_REQ_POST}', post_data)
        if resp['success']:
            return True, resp['description']

        return False, resp['description']
