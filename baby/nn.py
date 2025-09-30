from .tensor import Tensor 
from .ops import * 

class Parameter(Tensor):
    """
    A special kind of Tensor that is registered as a model parameter.

    When you assign a Parameter to an attribute of a Module, it is
    automatically added to the Module's list of parameters, and will
    be returned by the `parameters()` method.
    """
    pass


class Module:
    """
    The base class for all neural network modules (layers).

    Your models should also subclass this class. Modules can contain
    other Modules, allowing you to nest them in a tree structure.
    Ex: Linear(Module), Flatten(Module), Relu(Module)...

    You can assign Module instances as regular attributes:

    import baby.nn as nn
    
    class BabyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 20)
            self.layer2 = nn.Linear(20, 5)

        def forward(self, x):
            x = self.layer1(x)
            x = ops.relu(x)
            x = self.layer2(x)
            return x
    """
    def __call__(self, *args, **kwargs):
        """
        Makes the module callable (e.g., `model(input_data)`).
        This automatically calls the `forward` method.
        """

        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def parameters(self):
        """
        Returns a list of all Parameter objects in the module and its submodules.
        """
        return _get_params(self.__dict__)
    
def _get_params(value):
    """ 
        Returns all Parameter Objects.
    """
    if isinstance(value, Parameter):
        return [value]
    
    params = []

    if isinstance(value, Module):
        params +=  value.parameters()
    
    elif isinstance(value, dict):
        for v in value.values():
            params += _get_params(v)

    elif isinstance(value, (list, tuple)):
        for v in value:
            params+= _get_params(v)

    return params

 
    

    
