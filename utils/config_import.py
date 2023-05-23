import importlib
import os


def load_config_module(file_name):
    file_name = os.path.normpath(file_name)

    if file_name[-3:] == ".py":
        module_name = file_name[:-3]
    else:
        module_name = file_name

    module_name = module_name.replace("/", ".")
    config_module = importlib.import_module(module_name)

    return config_module


def import_localization_class(class_path):
    path = class_path.split(".")
    class_name = path[-1]
    module_path = ".".join(path[:-1])
    module = importlib.import_module(module_path)

    localization_class = getattr(module, class_name)

    return localization_class