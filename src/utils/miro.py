import time

import requests


class MiroHandler:
    """
    miro handler
    """

    def __init__(self, access_token, board_id, time_wait=1.0):
        self.access_token = access_token
        self.board_id = board_id
        self.time_wait = time_wait

    def _create_miro_object(self, data, miro_object_type="widgets"):
        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        url_create_widget = "https://api.miro.com/v1/boards/{}/{}".format(
            self.board_id, miro_object_type
        )
        response = requests.post(url_create_widget, json=data, headers=headers)
        print(response.text)
        time.sleep(self.time_wait)
        return response.text

    def add_sticky(self, text):
        data = self.build_sticker_data(text)
        return self._create_miro_object(data, miro_object_type="widgets")

    def build_sticker_data(self, text):
        data = {}
        data["type"] = "sticker"
        data_style = {}
        data_style["fontSize"] = 40
        data["style"] = data_style
        data["text"] = f"<p>{text}</p>"
        return data
