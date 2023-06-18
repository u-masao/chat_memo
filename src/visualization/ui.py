import json
import logging
import re
import time
from typing import Dict

import click
import gradio as gr
import mlflow
import openai
import pandas as pd
import yaml

from src.utils.miro import MiroHandler


def load_credential(credential_path: str) -> Dict:
    with open(credential_path) as fo:
        credential = yaml.safe_load(fo)
    return credential


def greet(text):
    BOARD_ID = "uXjVM9oIaSw="
    credential = load_credential("config/credential.yaml")
    access_token = credential["miro"]["access_token"]
    miro = MiroHandler(access_token=access_token, board_id=BOARD_ID)
    miro.add_sticky(text)
    return "Hello " + text + "!"


def define_functions():
    functions = [
        {
            "name": "put_sticky_to_miro",
            "description": "指定のメッセージを書いた付箋を miro のボードに貼り付ける",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "付箋に記述するメッセージ",
                    },
                },
                "requied": ["message"],
            },
        }
    ]

    return functions


def build_prompt():
    # query to openai
    instruction = """あなたは転職を希望する会社員です。
    ネガティブな転職理由を箇条書きにしなさい。"""
    # それぞれの転職理由に続いてポジティブな言い換えを書きなさい。

    output_format = """
    ### 出力する際のフォーマット
    - 10個から20個の転職理由を書くこと
    - 日本語で回答を記述すること
    - miro に付箋を貼ること

    ### 出力フォーマットの例
    ```
    - 転職理由
    - 転職理由
    - 転職理由
    - 転職理由
    ```

    ### 禁止事項
    - 同じ転職理由を書くことを禁ずる
    """
    # - ネガティブな転職理由とポジティブな言い換えは同じ行に書きなさい
    # - ポジティブな言い換えはカッコ内に書きなさい

    prompt = f"{instruction}\n{output_format}"
    return prompt


def parse_response(response):
    # init log
    logger = logging.getLogger(__name__)
    logger.info(f"full response: {response}")

    results = []
    for index, choice in enumerate(response.choices):
        texts = []
        if choice["finish_reason"] == "function_call":
            if (
                choice["message"]["function_call"]["name"]
                == "put_sticky_to_miro"
            ):
                # function calling の引数を取得
                arguments = choice["message"]["function_call"]["arguments"]
                logger.info(f"arguments: \n{arguments}")

                # json 文字列を dict に変換
                arguments_dict = json.loads(arguments)

                # parsing
                for text in arguments_dict["message"].split("\n"):
                    text = re.sub("^- ", "", text.strip())
                    logger.info(f"text: \n{text}")
                    if len(text) > 0:
                        texts.append(text)
                results.append(pd.DataFrame({"index": index, "text": texts}))
        else:
            logger.info(choice)

    result_df = pd.concat(results).reset_index(drop=True)
    return result_df


def generate_texts(openai_api_key, kwargs):
    # init log
    logger = logging.getLogger(__name__)
    # build prompt
    prompt = build_prompt()
    logger.info(f"prompt: \n{prompt}")

    # define functions
    functions = define_functions()
    logger.info(f"functions: \n{functions}")

    # APIリクエストの設定
    openai.api_key = openai_api_key
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": prompt}],
        functions=functions,
        function_call="auto",
        n=kwargs["param_n"],
    )
    elapsed_time = time.time() - start_time

    # logging
    logger.info(f"elapsed_time: {elapsed_time}")
    mlflow.log_metric("elapsed_time", elapsed_time)
    mlflow.log_metrics(response.usage)
    mlflow.log_param("model", response.model)
    n_choices = len(response.choices)
    mlflow.llm.log_predictions(
        [""] * n_choices, response.choices, [prompt] * n_choices
    )

    # log response dump
    openai_response_filepath = "data/interim/openai_response.json"
    json.dump(response, open(openai_response_filepath, "w"))
    mlflow.log_artifact(openai_response_filepath)

    # parse response
    result_df = parse_response(response)
    return prompt, result_df


def stick_to_miro(prompt, result_df, access_token, board_id):
    # init log
    logger = logging.getLogger(__name__)

    # init miro handler
    miro = MiroHandler(access_token=access_token, board_id=board_id)

    # add sticky of prompt
    miro.add_sticky(prompt)

    # add sticky of each text
    sorted_df = (
        result_df.groupby("text").count()[["index"]].sort_index().reset_index()
    )
    sorted_df["ratio"] = 100.0 * sorted_df["index"] / len(sorted_df)
    for index, row in sorted_df.iterrows():
        message = f"{index:03d}. {row['text']}({row['ratio']:0.1f})"
        miro.add_sticky(message)
        logger.info(message)

    sorted_output_filepath = "data/interim/miro_output.csv"
    sorted_df.to_csv(sorted_output_filepath)
    mlflow.log_artifact(sorted_output_filepath)


@click.command()
@click.argument("board_id", type=str)
@click.argument("output_filepath", type=click.Path())
@click.option("--param_n", type=int, default=10)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    # init log
    logger = logging.getLogger(__name__)
    logger.info("start process")
    logger.info({f"args.{k}": v for k, v in kwargs.items()})
    mlflow.set_experiment("転職理由をmiroへ貼る")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load credentials
    credential = load_credential("config/credential.yaml")

    # generate texts
    prompt, result_df = generate_texts(credential["openai"]["api_key"], kwargs)

    # output to csv
    result_df.to_csv(kwargs["output_filepath"])

    # output to miro
    stick_to_miro(
        prompt,
        result_df,
        access_token=credential["miro"]["access_token"],
        board_id=kwargs["board_id"],
    )

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
else:
    ui = gr.Interface(fn=greet, inputs="text", outputs="text")
    ui.launch(server_port=3000)
