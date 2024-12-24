from .carrot import Carrot
from .optim import optim
import pickle


def load(file_name):
    with open(file_name, "rb") as f:
        model_parameters_dict = pickle.load(f)
    return model_parameters_dict


def save(model_parameters_dict, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(model_parameters_dict, f)
