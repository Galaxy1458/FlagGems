from .ops import *  # noqa: F403


def get_register_config():
    FORWARD = False
    # BACKEND = True
    return {
        "add.Tensor": [add, FORWARD],
    }
