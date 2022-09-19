import os
import warnings


if os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "brocolli"
):
    message = (
        "You are importing brocolli within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "brocolli project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))
