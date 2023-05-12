import pandas as pd
import numpy as np
from sklearn import linear_model
import os

from config import properties as p
from src.models.model_main import Models


class LSTM(Models):

    def __init__(self):
        super().__init__()

