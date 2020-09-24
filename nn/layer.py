from abc import ABCMeta, abstractmethod

class AbstractLayer():
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def get_activation_grad(self, z, upstream_gradient):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, layer_err):
        # get downstream-gradients
        raise NotImplementedError()

    @abstractmethod
    def get_weight_grad(self, inputs, layer_err):
        # get gradients for weight update
        raise NotImplementedError()

    @abstractmethod
    def update(self, grad, lr):
        raise NotImplementedError()


