# python-background-remove

## Index

- [python-background-remove](#python-background-remove)
  - [Index](#index)
    - [Introduction](#introduction)
    - [Installation](#installation)
  

### Introduction

- Supports version of Python i.e. Python 3.10.1  along with Django 3.2.13 :zap:
- This project will first seperate the video frames and remove the background from the frames.
- To remove the background from frames we are using a DeepLabV3+ model trained on the human image segmentation dataset.
- For merging the background removed frames we have used the ffmpeg.

### Installation

> ##### 1. Clone repository

```sh
git clone 
```

> ##### 2. Create a virtual environment and activate it

```sh
python -m venv your-env-name
```

> For windows:
```sh
your-env-name\Script\activate
```

> For linux or Mac:
```sh
source your-env-name/bin/activate
```


> ##### 3. Install the dependencies
```sh
pip install -r requirements.txt
```

> #### 5. Start the server
```sh
    python manage.py runserver
```
<br />