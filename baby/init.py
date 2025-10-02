import math 
from .tensor import Tensor 

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0):
    """
    Xavier uniform initialization.
    Calls Tensor.rand() with the correct bounds.
    """
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return Tensor.rand(fan_in, fan_out, low=-a, high=a)
def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0):
    """
    Xavier normal initialization.
    Calls Tensor.randn() with the correct standard deviation.
    """
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return Tensor.randn(fan_in, fan_out, mean=0, std=std)


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu"):
    """
    Kaiming uniform initialization.
    Calls Tensor.rand() with the correct bounds.
    """
    bound = math.sqrt(2.0) * math.sqrt(3.0 / fan_in)
    return Tensor.rand(fan_in, fan_out, low=-bound, high=bound)


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu"):
    """
    Kaiming normal initialization.
    Calls Tensor.randn() with the correct standard deviation.
    """
    std = math.sqrt(2.0 / fan_in)
    return Tensor.randn(fan_in, fan_out, mean=0, std=std)