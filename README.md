# Segmentation-AppHistory
Provide Segmentation for  groups of individual Users using AppHistory data.

Has 2 possible models used: 

First is  LDA model

```bash
poetry shell
```

```bash
python segmentation_2.py
```

This one outputs an interactable HTML file with the topic segmentation for the users.

script_path,'ldavis_prepared_{num_topics).html'


Second is BERT model

```bash
poetry shell
```

```bash
python segmentation_BERT.py
```

This one outputs many different interactable HTML files with the topic segmentation for the users.

File names are output to terminal

## Installation

You will need [poetry](https://python-poetry.org/docs/) to install the dependencies of this project

## osx / linux / bashonwindows install instructions

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
```

Locate where poetry is installed

```bash
whereis poetry
```

Copy and replace poetry's path to this line and added it at the end of the `.bashrc` file

```bash
export PATH="$HOME/.poetry/bin:$PATH"
```

Check if poetry is already installed

```bash
poetry --version
```

## Install dependencies

```bash
poetry install
```

## The script needs SQLITE3 to run

On CENTOS:

```bash
yum install -y gcc make sqlite-devel zlib-devel libffi-devel openssl-devel wget 
```




## Necessary environment variables:

```
DB_USER_CITUS = "USERNAME"
DB_PASSWORD_CITUS = "12321312"
DB_NAME_CITUS = "cota"
DB_PORT_CITUS = "1111"
DB_HOST_CITUS = "11.11.111.213"


AMOUNT_TO_EXTRACT=500
TOPICS_TO_MODEL=5
```

