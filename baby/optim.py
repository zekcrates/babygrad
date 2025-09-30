
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
