
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

    def zero_grad(self):
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
    This optimizer updates parameters by taking a step in the direction of the
    negative gradient, scaled by the learning rate.
    """
    def __init__(self, params, lr=0.01):
        """
        Args:
            params (list[Parameter]): A list of Tensors to optimize, typically
                                   from model.parameters().
            lr (float, optional): The learning rate for the optimizer. 
                                  Defaults to 0.01.
        """
        super().__init__(params)
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step.
        """
        for param in self.params:
            if param.grad is not None:
                # Note: The update is performed on the .data attribute
                param.data -= self.lr * param.grad

class Adam(Optimizer):
    """
    Implements the Adam optimization algorithm.
    """
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        """

        Args:
            params (list[Tensor]): A list of Tensors to optimize.(eg: model.parameters())
            lr (float, optional): The learning rate.
            beta1 (float, optional): The exponential decay rate for the
                                     first moment estimates. Defaults to 0.9.
            beta2 (float, optional): The exponential decay rate for the
                                     second-moment estimates. Defaults to 0.999.
            eps (float, optional): A small constant for numerical stability.
                                   Defaults to 1e-8.
            weight_decay (float, optional): L2 penalty (weight decay).
                                            Defaults to 0.0.
        """
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0 

        self.m = {}
        self.v = {} 

    def step(self):
        
        self.t += 1
        for param in self.params:
            if param.grad is not None:
                grad = param.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param.data
                mt = self.m.get(param, 0) * self.beta1 + (1 - self.beta1) * grad
                self.m[param] = mt
                vt = self.v.get(param, 0) * self.beta2 + (1 - self.beta2) * (grad ** 2)
                self.v[param] = vt
                mt_hat = mt / (1 - (self.beta1 ** self.t))

                vt_hat = vt / (1 - (self.beta2 ** self.t))

                denom = (vt_hat**0.5 + self.eps)
                param.data -= self.lr * mt_hat / denom