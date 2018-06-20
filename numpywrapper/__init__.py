import json
import os

file_name = fn = os.path.join(os.path.dirname(__file__), 'use_gpu.json')


def set_use_gpu(value):
    with open(file_name, "w+") as f:
        json.dump({"use_gpu": value}, f)


def get_use_gpu():
    with open(file_name) as f:
        use_gpu = json.load(f)["use_gpu"]
    return use_gpu


use_gpu = get_use_gpu()

if use_gpu:
    set_use_gpu(False)
    from .cupy_wrapper import *
else:
    set_use_gpu(False)
    from .numpy_wrapper import *

