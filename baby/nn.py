from .tensor import Tensor


class Parameter(Tensor):
    """
    A special Tensor that tells a Module it is a learnable parameter.

    Think of this as a regular Tensor that has been "marked" as a weight or
    bias that should be updated by the optimizer during training. When you
    create a model, you should wrap all learnable tensors (like the weights
    and biases of a linear layer) in this class.

    Example:
        >>> # A regular tensor attribute - will be ignored by model.parameters()
        >>> self.some_data = Tensor([1, 2, 3])
        >>>
        >>> # A parameter - will be found and trained by the optimizer!
        >>> self.weights = Parameter(Tensor.randn(10, 5))
    """
    pass

def _get_parameters(obj) -> list[Parameter]:
    """
    A simple recursive function that finds all Parameter objects within any given
    object by searching through its attributes, lists, tuples, and dicts.
    """
    params = []
    
    if isinstance(obj, Parameter):
        return [obj]
    
    if isinstance(obj, Module):
        return obj.parameters()

    if isinstance(obj, dict):
        for value in obj.values():
            params.extend(_get_parameters(value))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            params.extend(_get_parameters(item))

    return params

def _get_modules(obj) -> list['Module']:
    """
    A simple recursive function that finds all Module objects within any given
    object by searching through its attributes, lists, tuples, and dicts.
    """
    modules = []
    
    if isinstance(obj, Module):
        return [obj]
        
    if isinstance(obj, dict):
        for value in obj.values():
            modules.extend(_get_modules(value))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            modules.extend(_get_modules(item))

    return modules


class Module:
    """
    The base class for all neural network modules (layers).
    """
    def __init__(self):
        self.training = True

    def parameters(self) -> list[Parameter]:
        """
        Returns a list of all parameters in the module and its submodules.
        """
        params = _get_parameters(self.__dict__)        
        unique_params = []
        seen_ids = set()
        for p in params:
            if id(p) not in seen_ids:
                unique_params.append(p)
                seen_ids.add(id(p))
        return unique_params

    def train(self):
        """Sets this module and all its submodules to training mode."""
        self.training = True
        # Find all child modules and recursively call train() on them.
        for m in _get_modules(self.__dict__):
            m.train()

    def eval(self):
        """Sets this module and all its submodules to evaluation mode."""
        self.training = False
        # Find all child modules and recursively call eval() on them.
        for m in _get_modules(self.__dict__):
            m.eval()

    def forward(self, *args, **kwargs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        Makes the module callable like a function.

        This is a Python "magic method" that allows you to treat your module
        instance as a function. It automatically calls the `forward` method
        that you defined.

        Example:
            >>> model = MyModel(10, 2)
            >>> input_tensor = Tensor.randn(64, 10)
            >>> output = model(input_tensor)  # This calls model.forward(input_tensor)
        """
        return self.forward(*args, **kwargs)
