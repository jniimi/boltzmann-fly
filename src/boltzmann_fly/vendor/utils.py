import os, re, json, pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch

def find_device() -> torch.device:
    """
    利用可能なデバイスを探索して返す。
    優先順位: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def check_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be `dict` type.")
    for key in ["dataset", "bm", "gpt", "adapter"]:
        if not key in config.keys():
            raise ValueError(f"key `{key}` is not found in config.")
    return config

def send_message(message, web_hook_url=None):
    if not web_hook_url is None:
        WEB_HOOK_URL = web_hook_url
    else:
        WEB_HOOK_URL = os.getenv('SLACK_WEBHOOK_URL', None)
    if WEB_HOOK_URL is None:
        raise ValueError("Cannot obtain Slack's webhook url.")
    import requests  # boltzmann-fly: lazy import
    requests.post(WEB_HOOK_URL, data=json.dumps({
        'text': message
    }))