# Benchmarking LeanAide

To evaluate and benchmark LeanAide, we will test and evaluate on presently known various datasets.

## Datasets

- [FATE](https://github.com/frenzymath/FATE) - consists of 3 datasets, FATE-H, FATE-M, FATE-X

## Setup

To setup the environment, you can use [uv](https://docs.astral.sh/uv/getting-started/installation/)(or any other environment manager):

```bash
uv sync
```

and then you can source the environment with

```bash
source .venv/bin/activate
```

## Run the script

After creating the environment, you can open [benchmark.py](./benchmark.py), add details of JSON file to be evaluated, and you can get the results.

> [!IMPORTANT]
> The benchmark assumes that the backend UI and server BOTH are running on the specific URL's.

The benchmark calls the UI API calls for LLM Responses and Document JSON generation(you can write it manually but the above script uses prewritten code for the same). The result json will initially contain Tokens for the problems, which can later on be rerun to be updated. When you rerun the same file, if the AI responses are present, it won't rerun them and it will check the status of Token and update accordingly.
