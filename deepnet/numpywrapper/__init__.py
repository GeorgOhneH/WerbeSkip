import json


def set_use_gpu(value):
    with open("numpywrapper/use_gpu.json", "w+") as f:
        json.dump({"use_gpu": value}, f)


def get_use_gpu():
    with open("numpywrapper/use_gpu.json") as f:
        use_gpu = json.load(f)["use_gpu"]
    return use_gpu


use_gpu = get_use_gpu()

if use_gpu:
    set_use_gpu(False)
    from .cupy_wrapper import *
else:
    set_use_gpu(False)
    from .numpy_wrapper import *

