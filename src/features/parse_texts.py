import json
import logging

import click
import mlflow
import pandas as pd


def parse_response(response):
    # init log
    logger = logging.getLogger(__name__)
    logger.info(f"full response: {response}")

    results = []
    for index, choice in enumerate(response["choices"]):
        if choice["finish_reason"] == "function_call":
            if choice["message"]["function_call"]["name"] == "create_csv_file":
                # function calling の引数を取得
                arguments = choice["message"]["function_call"]["arguments"]
                # json 文字列を dict に変換
                messages = json.loads(arguments)["text"]

        elif "message" in choice:
            messages = choice["message"]["content"]

        # parsing
        column_names = ["ネガティブな転職理由", "ポジティブな言い換え", "カテゴリ"]
        lines = []
        for line in messages.split("\n"):
            line = line.strip()
            logger.info(f"line : \n{line }")
            if len(line) > 0:
                cells = line.split(",")
                if len(cells) == 3 and cells[1] != column_names[1]:
                    lines.append(cells)
        result_df = pd.DataFrame(lines, columns=column_names).assign(
            index=index
        )
        results.append(result_df)

    result_df = pd.concat(results).reset_index(drop=True)
    return result_df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """メイン処理"""

    # init log
    logger = logging.getLogger(__name__)

    # logging
    logger.info("start process")
    logger.info({f"args.{k}": v for k, v in kwargs.items()})
    mlflow.set_experiment("parse_texts")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    response = json.load(open(kwargs["input_filepath"], "r"))
    result_df = parse_response(response)
    result_df.to_csv(kwargs["output_filepath"])

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
