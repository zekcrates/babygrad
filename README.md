# Simple Deep learning library 

This is a simple(baby) implementation of an autograd engine that supports `Tensors`. 


```python 

from baby import Tensor 

a = Tensor([1,2,3])
b = Tensor([3,4,5])
c = a+b 
d = Tensor([4,5,6])
e = c*d 
print(e)
print(e.__dict__)
e.backward()
print(e.grad)

```

# Setup 

```r
git clone <REPO>
cd <REPO>

#Create virtual environment
#we use uv package manager 

uv pip install numpy pytest 
```

# Tests

```r
# wsl2/linux 
PYTHONPATH=. uv run pytest 

# windows 
$env:PYTHONPATH="."   
uv run pytest 
```

