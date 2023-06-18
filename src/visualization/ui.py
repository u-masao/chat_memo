import logging
import random
import time
from typing import Dict

import click
import gradio as gr
import requests
import yaml


def load_credential(credential_path: str) -> Dict:
    with open(credential_path) as fo:
        credential = yaml.safe_load(fo)
    return credential


class MiroHandler:
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


def greet(text):
    BOARD_ID = "uXjVM9oIaSw="
    credential = load_credential("config/credential.yaml")
    access_token = credential["miro"]["access_token"]
    miro = MiroHandler(access_token=access_token, board_id=BOARD_ID)
    miro.add_sticky(text)
    return "Hello " + text + "!"


def randstr(length):
    return "".join([chr(random.randint(97, 122))] * length)


@click.command()
@click.argument("board_id", type=str)
def main(**kwargs):
    logger = logging.getLogger(__name__)
    logger.info("start process")
    credential = load_credential("config/credential.yaml")
    access_token = credential["miro"]["access_token"]
    miro = MiroHandler(access_token=access_token, board_id=kwargs["board_id"])
    [miro.add_sticky(randstr(10)) for _ in range(10)]
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
else:
    ui = gr.Interface(fn=greet, inputs="text", outputs="text")
    ui.launch(server_port=3000)
