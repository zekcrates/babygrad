
class Optimizer:
    """
    Base class for all optimizers.

    This class defines the basic interface for an optimizer. It holds the
    parameters to be updated and provides methods to step the optimization
    and reset gradients.

    Example of a subclass:
        class SGD(Optimizer):
            def __init__(self, params, lr=0.01):
                super().__init__(params)
                self.lr = lr

            def step(self):
                pass
    """
    def __init__(self, params):
        """
        Initializes the optimizer.

        Args:
            params (list[Parameter]): A list of parameters to optimize,
                                      typically from model.parameters().
        """
        self.params = params

    def reset_grad(self):
        """
        Sets the gradients of all optimized parameters to None.
        This is typically called at the start of each training iteration.
        """
        for p in self.params:
            p.grad = None

    def step(self):
        """
        Performs a single optimization step (e.g., updating parameters).
        This method must be implemented by a subclass.
        """
        raise NotImplementedError

class SGD(Optimizer):
    """
    Implements the Stochastic Gradient Descent (SGD) optimizer.

    This optimizer updates parameters by taking a step in the direction of the
    negative gradient, scaled by the learning rate.
    """
    def __init__(self, params, lr=0.01):
        """
        Initializes the SGD optimizer.

        Args:
            params (list[Tensor]): A list of Tensors to optimize, typically
                                   from model.parameters().
            lr (float, optional): The learning rate for the optimizer. 
                                  Defaults to 0.01.
        """
        super().__init__(params)
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.

        This method iterates through the parameters and updates their data
        in-place using the standard SGD update rule:
        param.data = param.data - learning_rate * param.grad
        """
        for param in self.params:
            if param.grad is not None:
                # Note: The update is performed on the .data attribute
                param.data -= self.lr * param.grad


