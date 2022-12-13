from layers import *
import numpy as np
import time


class CNN:
    def __init__(self):
        self.layers = []
        self.info_df = {}
        self.parameters = []
        self.optimizer = ""
        self.loss = "mse"
        self.lr = 0.01
        self.mr = 0.0001
        self.metrics = []
        self.av_optimizers = ["sgd", "momentum", "adam"]
        self.av_metrics = ["mse", "accuracy", "cse"]
        self.av_loss = ["mse", "cse"]
        self.iscompiled = False
        self.model_dict = None
        self.out = []
        self.eps = 1e-15
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}

    def add(self, layer):
        pass
