import pytest
import numpy as np

from baby.tensor import Tensor
from baby.nn import Module, Parameter


class SimpleLayer(Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(np.random.randn(5, 5))

class ComplexModel(Module):
    
    def __init__(self):
        super().__init__()
        # 1. A direct parameter
        self.p_direct = Parameter(np.random.randn(10))
        self.layer_direct = SimpleLayer()
        self.layer_list = [SimpleLayer(), SimpleLayer()]
        self.layer_dict = {"a": SimpleLayer(), "b": SimpleLayer()}
        
        self.not_a_parameter = Tensor(np.random.randn(3, 3))
        self.shared_param = self.layer_direct.weight


def test_parameter_discovery():
    """
    Tests that the .parameters() method correctly finds all unique, nested parameters.
    """
    model = ComplexModel()

    params = model.parameters()
    assert len(params) == 6, "Incorrect number of unique parameters found."

    expected_param_ids = {
        id(model.p_direct),
        id(model.layer_direct.weight),
        id(model.layer_list[0].weight),
        id(model.layer_list[1].weight),
        id(model.layer_dict["a"].weight),
        id(model.layer_dict["b"].weight),
    }
    
    returned_param_ids = {id(p) for p in params}
    
    assert expected_param_ids == returned_param_ids, "The set of found parameters is incorrect."


def test_train_eval_mode_switching():
    """
    Tests that .train() and .eval() modes are correctly cascaded to all submodules.
    """
    model = ComplexModel()
    
    all_modules = [
        model,
        model.layer_direct,
        model.layer_list[0], model.layer_list[1],
        model.layer_dict["a"], model.layer_dict["b"],
    ]
    
    assert all(m.training for m in all_modules), "Modules should initialize in training mode."

    model.eval()
    
    assert not any(m.training for m in all_modules), "model.eval() did not switch all submodules to eval mode."
    
    model.train()

    assert all(m.training for m in all_modules), "model.train() did not switch all submodules back to train mode."