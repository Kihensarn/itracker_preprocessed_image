import os
import json
import torch
import yaml
import numpy as np
import shutil
from pathlib import Path


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def save_model(state, is_best, path):
    model_dir = Path(path)
    if not model_dir.is_dir():
        model_dir.mkdir(parents=True)

    model_path = model_dir.joinpath('model_{}.pth.tar'.format(state['epoch']))
    if not model_path.is_file():
        torch.save(state, str(model_path))

    if is_best:
        best_model_path = model_dir.joinpath('best_model_{}.pth.tar'.format(state['epoch']))
        shutil.copyfile(str(model_path), str(best_model_path))


def load_model(model_name, path):
    model_dir = Path(path)
    if not model_dir.is_dir():
        print('invalidate model path')
        return None
    model_path = model_dir / model_name
    if not model_path.is_file():
        print('not such a model')
        return None
    state = torch.load(str(model_path), map_location='cpu')
    return state


def load_configs(yaml_fn):
    with open(yaml_fn, 'r') as f:
        return yaml.load(f)


def save_results(res):
    np.savetxt('within_eva_results.txt', res, delimiter=',')

    if os.path.exists('submission_within_eva.zip'):
        os.system('rm submission_within_eva.zip')

    os.makedirs('submission_within_eva')
    os.system('mv within_eva_results.txt ./submission_within_eva')
    os.system('zip -r submission_within_eva.zip submission_within_eva')
    os.system('rm -rf submission_within_eva')