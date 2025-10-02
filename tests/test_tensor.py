import numpy as np
import pytest
from baby.tensor import Tensor

def test_tensor_creation_from_list():
    t = Tensor([1, 2, 3])
    assert isinstance(t.data, np.ndarray), "Data should be converted to a numpy array."
    assert t.dtype == np.float32, "Default dtype should be float32 for lists."
    assert np.array_equal(t.data, np.array([1, 2, 3]))

def test_tensor_creation_from_numpy():
    arr_int = np.array([4, 5, 6], dtype=np.int32)
    t1 = Tensor(arr_int)
    assert t1.dtype == np.float32, "Should cast numpy array to default float32."
    
    t2 = Tensor(arr_int, dtype=None)
    assert t2.dtype == np.int32, "dtype=None should preserve original numpy dtype."

    t3 = Tensor(arr_int, dtype="float64")
    assert t3.dtype == np.float64, "Should be able to cast to a specified dtype."

def test_tensor_creation_from_tensor_copy():
    
    t1 = Tensor([1, 2, 3], requires_grad=False)

    t2 = Tensor(t1, requires_grad=True)

    assert isinstance(t2, Tensor)
    assert np.array_equal(t1.data, t2.data), "Data values should be the same."
    
    assert id(t1.data) != id(t2.data), "Data should be copied, not referenced."
    
    assert t1.requires_grad is False, "Original tensor's properties should not change."
    assert t2.requires_grad is True, "New tensor should have its own properties."

def test_tensor_creation_from_tensor_cast_dtype():
    
    t1_f32 = Tensor([10, 20, 30], dtype="float32")
    
    t2_f64 = Tensor(t1_f32, dtype="float64")
    assert t1_f32.dtype == np.float32
    assert t2_f64.dtype == np.float64
    
    t3_f32 = Tensor(t1_f32, dtype=None)
    assert t3_f32.dtype == np.float32

def test_parameter_creation_from_tensor():
  
    from baby.nn import Parameter 
    
    t = Tensor([5, 6, 7])
    p = Parameter(t)
    
    assert isinstance(p, Parameter)
    assert isinstance(p, Tensor)
    assert np.array_equal(p.data, t.data)
    assert id(p.data) != id(t.data)