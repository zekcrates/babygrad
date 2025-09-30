# Simple Deep learning library 

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