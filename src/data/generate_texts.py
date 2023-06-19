import json
import logging
import time
from typing import Dict, List

import click
import mlflow
import openai
import yaml


def load_credential(credential_path: str) -> Dict:
    """credential ファイルを読み込む"""
    with open(credential_path) as fo:
        credential = yaml.safe_load(fo)
    return credential


def define_functions() -> List[Dict]:
    """chatgpt 向けの function calling を定義"""
    functions = [
        {
            "name": "create_csv_file",
            "description": "CSV 形式のファイルを作成する",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "ファイルの内容",
                    },
                },
                "requied": ["text"],
            },
        }
    ]

    return functions


def build_prompt() -> str:
    """
    プロンプトを作成
    """
    # query to openai
    prompt = """
あなたは転職を希望する会社員です。
転職理由を箇条書きにしなさい。

### 出力する際のフォーマット
- csv 形式で出力すること
- 日本語で回答を記述すること
- 10個から20個のネガティブな転職理由を書くこと
- ポジティブな言い換えを書くこと
- 転職理由をカテゴリ分類すること

### 出力フォーマットの例
```
ネガティブな転職理由,ポジティブな言い換え,カテゴリ
ネガティブな転職理由,ポジティブな言い換え,カテゴリ
ネガティブな転職理由,ポジティブな言い換え,カテゴリ
ネガティブな転職理由,ポジティブな言い換え,カテゴリ
```

### 禁止事項
- 同じ転職理由を書くことを禁ずる
    """
    return prompt


def generate_texts(openai_api_key, kwargs):
    """テキストを生成"""

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
    mlflow.log_param("n_choices", len(response.choices))

    return prompt, response


@click.command()
@click.argument("output_generated_filepath", type=click.Path())
@click.argument("output_prompt_filepath", type=click.Path())
@click.option("--param_n", type=int, default=10)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """メイン処理"""

    # init log
    logger = logging.getLogger(__name__)

    # logging
    logger.info("start process")
    logger.info({f"args.{k}": v for k, v in kwargs.items()})
    mlflow.set_experiment("転職理由を生成する")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # load credentials
    credential = load_credential("config/credential.yaml")

    # generate texts
    prompt, response = generate_texts(credential["openai"]["api_key"], kwargs)

    # log response dump
    openai_response_filepath = kwargs["output_generated_filepath"]
    json.dump(response, open(openai_response_filepath, "w"))
    mlflow.log_artifact(openai_response_filepath)
    prompt_filepath = kwargs["output_prompt_filepath"]
    print(prompt, file=open(prompt_filepath, "w"))
    mlflow.log_artifact(prompt_filepath)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
