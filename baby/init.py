# import math 
# from .tensor import Tensor 

# def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs):
#     """
#     Xavier uniform initialization.
#     Calls Tensor.rand() with the correct bounds.
#     """
#     a = gain * math.sqrt(6.0 / (fan_in + fan_out))
#     return Tensor.rand(fan_in, fan_out, low=-a, high=a)
# def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs):
#     """
#     Xavier normal initialization.
#     Calls Tensor.randn() with the correct standard deviation.
#     """
#     std = gain * math.sqrt(2.0 / (fan_in + fan_out))
#     return Tensor.randn(fan_in, fan_out, mean=0, std=std)


# def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs):
#     """
#     Kaiming uniform initialization.
#     Calls Tensor.rand() with the correct bounds.
#     """
#     bound = math.sqrt(2.0) * math.sqrt(3.0 / fan_in)
#     return Tensor.rand(fan_in, fan_out, low=-bound, high=bound)


# def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs):
#     """
#     Kaiming normal initialization.
#     Calls Tensor.randn() with the correct standard deviation.
#     """
#     std = math.sqrt(2.0 / fan_in)
#     return Tensor.randn(fan_in, fan_out, mean=0, std=std)
import math 
from .tensor import Tensor 

def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs):
    """
    Xavier uniform initialization.
    """
    # Remove device from kwargs if present (not supported by Tensor.rand)
    kwargs.pop('device', None)
    
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    
    if shape is None:
        shape = (fan_in, fan_out)
    
    return Tensor.rand(*shape, low=-a, high=a, **kwargs)


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, shape=None, **kwargs):
    """
    Xavier normal initialization.
    """
    # Remove device from kwargs if present
    kwargs.pop('device', None)
    
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    if shape is None:
        shape = (fan_in, fan_out)
    
    return Tensor.randn(*shape, mean=0, std=std, **kwargs)


def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs):
    """
    Kaiming uniform initialization.
    """
    # Remove device from kwargs if present
    kwargs.pop('device', None)
    
    bound = math.sqrt(2.0) * math.sqrt(3.0 / fan_in)
    
    if shape is None:
        shape = (fan_in, fan_out)
    
    return Tensor.rand(*shape, low=-bound, high=bound, **kwargs)


def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs):
    """
    Kaiming normal initialization.
    """
    # Remove device from kwargs if present
    kwargs.pop('device', None)
    
    std = math.sqrt(2.0 / fan_in)
    
    if shape is None:
        shape = (fan_in, fan_out)
    
    return Tensor.randn(*shape, mean=0, std=std, **kwargs)


def rand(*shape, low=0.0, high=1.0, dtype="float32", requires_grad=True):
    """
    Helper function for uniform random initialization.
    """
    return Tensor.rand(*shape, low=low, high=high, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, dtype="float32", requires_grad=True):
    """
    Helper function to create a zero-initialized tensor.
    """
    return Tensor.zeros(*shape, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, dtype="float32", requires_grad=True):
    """
    Helper function to create a one-initialized tensor.
    """
    return Tensor.ones(*shape, dtype=dtype, requires_grad=requires_grad)