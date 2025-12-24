import os
import pickle
import numpy as np
from baby.nn import Linear, Sequential, ReLU
from baby.tensor import Tensor
def test_model_save_and_load():
    model = Sequential(
        Linear(5, 10),
        ReLU(),
        Linear(10, 2)
    )
    x = Tensor(np.random.randn(1, 5))
    original_output = model(x).data.copy()

    filename = "test_model.pkl"
    model.save(filename)
    
    new_model = Sequential(
        Linear(5, 10),
        ReLU(),
        Linear(10, 2)
    )
    
    new_output_before_load = new_model(x).data
    assert not np.allclose(original_output, new_output_before_load)

    new_model.load(filename)
    
    loaded_output = new_model(x).data
    assert np.allclose(original_output, loaded_output), "Loaded model weights do not match original!"
    
    if os.path.exists(filename):
        os.remove(filename)
    print("Persistence Test: PASSED")