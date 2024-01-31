import json


def saveJson(path, data):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def readJson(path):
    try:
        with open(path, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = {}
    return existing_data


def updateJson(path, new_data):
    existing_data = readJson(path)
    existing_data.update(new_data)
    saveJson(path, existing_data)
