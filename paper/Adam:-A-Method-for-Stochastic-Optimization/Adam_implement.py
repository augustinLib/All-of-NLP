import numpy as np
from collections import OrderedDict


class BaseOptimizer():
    def __init__(self, lr = 0.01):
        self.lr = lr
        
    def __repr__(self) -> str:
        return "Optimizer"
        

class Adam(BaseOptimizer):
    def __init__(self, lr=0.001, b1 = 0.9, b2 = 0.999, epsilon = 1e-8):
        super().__init__(lr)
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.iter = 0
    

    def update(self, model):
        if self.m is None:
            self.m, self.v = OrderedDict(), OrderedDict()
            for layer_name in model.sequence:
                layer = model.network[layer_name]
                if layer.differentiable:
                    self.m[layer_name], self.v[layer_name] = OrderedDict(), OrderedDict()

                    self.m[layer_name]["weight"] = np.zeros_like(layer.parameter["weight"])
                    self.m[layer_name]["bias"] = np.zeros_like(layer.parameter["bias"])

                    self.v[layer_name]["weight"] = np.zeros_like(layer.parameter["weight"])
                    self.v[layer_name]["bias"] = np.zeros_like(layer.parameter["bias"])

        self.iter += 1

        for layer_name in model.sequence:
            layer = model.network[layer_name]
            if layer.differentiable:
                # update m, v
                self.m[layer_name]["weight"] = self.b1 * self.m[layer_name]["weight"] + (1-self.b1) * layer.dw
                self.m[layer_name]["bias"] = self.b1 * self.m[layer_name]["bias"] + (1-self.b1) * layer.db
                self.v[layer_name]["weight"] = self.b2 * self.v[layer_name]["weight"] + (1-self.b2) * (layer.dw ** 2)
                self.v[layer_name]["bias"] = self.b2 * self.v[layer_name]["bias"] + (1-self.b2) * (layer.db ** 2)

                m_hat_weight = self.m[layer_name]["weight"] / (1 - (self.b1 ** self.iter))
                m_hat_bias = self.m[layer_name]["bias"] / (1 - (self.b1 ** self.iter))
                v_hat_weight = self.v[layer_name]["weight"] / (1 - (self.b2 ** self.iter))
                v_hat_bias = self.v[layer_name]["bias"] / (1 - (self.b2 ** self.iter))
                
                # update parameter
                model.network[layer_name].parameter["weight"] -= self.lr / (np.sqrt(v_hat_weight) + self.epsilon) * m_hat_weight
                model.network[layer_name].parameter["bias"] -= self.lr / (np.sqrt(v_hat_bias) + self.epsilon) * m_hat_bias