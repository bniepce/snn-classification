import logging, yaml


def check_config_file(config_file):
    """
    Checks basic structure of a config file.

    :param config_file: dict
        Config file to check.
    :return:
        config_file: dict
            Checked config file.
    """
    assert type(config_file) is dict, "Config file should be a dictionary"

    file_keys = list(config_file.keys())
    base_message = "[SNN-CLASSIFICATION::main::check_config_file (AssertionError)]: "
    modules = ["network"]

    for mdx, module in enumerate(modules):
        if mdx > 2:
            if module not in file_keys:
                config_file[module] = None
            continue
        assert module in file_keys, (
            base_message + 'Config file should contain keyword "' + module + '"!'
        )
    return config_file


def get_parameter_file(path):
    config = None

    try:
        stream_file = open(path, "r")
        config = yaml.load(stream_file, Loader=yaml.FullLoader)
        config = check_config_file(config)
        logging.info(
            "[{}::main] Success: Loaded configuration file at: {}".format(
                config["name"], path
            )
        )
    except:
        logging.error(
            "[{}::main] ERROR: Invalid configuration file at: {}, exiting...".format(
                config["name"], path
            )
        )
        exit()

    return config
