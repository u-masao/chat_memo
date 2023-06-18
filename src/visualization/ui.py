import logging
import time

import click
import gradio as gr
import requests
import yaml


def stick_note(text):
    with open("config/credential.yaml") as fo:
        credential = yaml.safe_load(fo)
    print(credential)
    access_token = credential["miro"]["access_token"]
    headers = {"Authorization": "Bearer {}".format(access_token)}
    BOARD_ID = "uXjVM9oIaSw="
    url_create_widget = "https://api.miro.com/v1/boards/{}/widgets".format(
        BOARD_ID
    )

    my_data = ["aaaa", "bbbb", "cccc"]
    for i, data_line in enumerate(my_data):
        data = {}

        data["type"] = "sticker"
        data["x"] = 100
        data["y"] = 0 + 114 * i
        # data["height"] = 228
        # data["width"] = 199

        data_style = {}
        data_style["backgroundColor"] = "#f5d128"
        # data_style["fontFamily"] = "Noto Sans"
        data_style["fontSize"] = 40
        data_style["textAlign"] = "left"
        data_style["textAlignVertical"] = "top"
        data["style"] = data_style
        text_in_sticker = data_line[0]
        data["text"] = "<p>{}</p>".format(text_in_sticker)

        response = requests.post(url_create_widget, json=data, headers=headers)

        print(response.text)

        time.sleep(1)


def greet(text):
    stick_note(text)
    return "Hello " + text + "!"


@click.command()
def main(**kwargs):
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    stick_note("aaaa")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
else:
    ui = gr.Interface(fn=greet, inputs="text", outputs="text")
    ui.launch(server_port=3000)
