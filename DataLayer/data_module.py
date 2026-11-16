from datetime import datetime
import json
import os


def create_dir(dir_path):
    """
    Checks if a directory exists. If not, create it.
    If it does exist, raises an exception.

    Args:
      dir_path (str): The path to the directory.
    """
    if not os.path.exists(dir_path):
        print(f"{dir_path} does not exist - creating it")
        os.makedirs(dir_path, exist_ok=False)
    else:
        raise Exception(f"{dir_path} already exists")


def init_date_dir(version_path, sub_dirs=()):
    """
    initialize a new experiment directory and return its path
    :param version_path: the version of the experiment.
    :param sub_dirs: a list of subdirectories to create
    :return: the path of the new experiment directory
    """
    time_now = datetime.now().strftime('%H%M-%d%m%Y')
    path = rf"{version_path}\{time_now}"
    os.makedirs(path, exist_ok=False)
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(path, sub_dir), exist_ok=False)
    return path, time_now


def delete_date_dir(date_path):
    """
    delete the experiment directory and all its contents and subdirectories
    :param date_path: the path of the experiment directory
    """
    if os.path.exists(date_path):
        for root, dirs, files in os.walk(date_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(date_path)


def list_files(root_dir, save_path=None):
    """
    Create a list of all files in the given root directory.
    Returns a dictionary of file paths and their last modified time.
    :param root_dir: The root directory to create a list of files from.
    :return: A list of file paths.
    """
    files = {}
    for dir_path, dir_names, file_names in os.walk(root_dir):
        for filename in file_names:
            file_path = os.path.join(dir_path, filename)
            # last modified time
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            files[os.path.join(dir_path, filename)] = file_mod_time.isoformat()

    if save_path is not None:
        save_dict(files, save_path)
    return files


def save_dict(dict_to_save, path, encoding=None):
    """
    Save the dict to a JSON file.
    :param dict_to_save: Dictionary to save.
    :param path: destination path.
    :param encoding: encoding of the file
    """
    if ".json" not in path:
        path += ".json"
    if encoding is not None:
        with open(path, "w", encoding=encoding) as f:
            json.dump(dict_to_save, f, indent=4)
    else:
        with open(path, "w") as f:
            json.dump(dict_to_save, f, indent=4)


def load_dict(path, encoding=None):
    """
    Load a dictionary from a JSON file.
    :param path: source path.
    :param encoding: encoding of the file
    :return: dictionary
    """
    if ".json" not in path:
        path += ".json"
    if not os.path.exists(path):
        raise Exception(f"{path} does not exist")
    if encoding is not None:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)
