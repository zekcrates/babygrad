


def print_graph(tensor, visited=None, depth=0):
    if visited is None:
        visited = set() 
    
    if id(tensor) in visited:
        return 
    
    visited.add(id(tensor))
    indent = "  " * depth

    if tensor._op is None:
        print(indent + "Leaf Tensor", tensor.shape)

    else:
        print(indent + f"{type(tensor._op).__name__} -> {tensor.shape}")

    for inp in tensor._inputs:
        print_graph(inp, visited, depth + 1)


def get_topo_order(tensor):
    topo= []
    visited = set()

    def walk(node):
        if id(node) in visited:
            return 
        visited.add(id(node))
        for inp in node._inputs:
            walk(inp)

        topo.append(node)

    walk(tensor)
    return topo 


from baby import Tensor

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)

c = a + b
d = c * 2
e = d.sum()

z = e



def execute(topo_order):
    computed = {}
    for node in topo_order:
        if node._op is None:
            computed[id(node)] = node 
        else:
            inputs = [computed[id(inp)] for inp in node._inputs]
            result = node._op.forward(*[inp.data for inp in inputs])

            new_tensor = Tensor(result, requires_grad=False)
            computed[id(node)] = new_tensor

    
    return computed[id(topo_order[-1])]



def compile(fn):
    traced = False 
    topo_order = None 

    def compiled_fn(*args):
        nonlocal traced,topo_order
        if not traced:

            output = fn(*args)
            topo_order = get_topo_order(output)
            traced =True 
        return execute(topo_order)
    return compiled_fn



def remove_identity_ops(topo):
    new = []

    replacement_map = {}

    for node in topo:
        if node._op  is None:
            new.append(node)
            continue
        op_name = type(node._op).__name__

        if op_name == "Add":
            inputs = node._inputs 
            
            if len(inputs) ==2:
                left,right = inputs 
                if right._op is None and hasattr(right,"data"):
                    if(right.data == 0 ).all():
                        print("Removing add")
                        
                        new.append(left)
                        
                        # for graph rewrite 
                        replacement_map[id(node)] = left 

                        continue

                if left._op is None and hasattr(left, "data"):
                    if (left.data == 0).all():
                        print("Removing Add (left is zero)")
                        replacement_map[id(node)] = right
                        continue
        
        if op_name == "MulScalar":
            node_input = node._inputs[0]
            scalar = node._op.scalar 
            if scalar == 1:
                replacement_map[id(node)] = node_input
                continue

            if scalar == 0 :
                zero_tensor = Tensor(0.0)
                replacement_map[id(node)] = zero_tensor
                continue
        else:
            new.append(node)

    return new , replacement_map




from baby import Tensor

x = Tensor([[1., 2., 3.]], requires_grad=True)     
W = Tensor([[1., 0.],                           
            [0., 1.],
            [1., 1.]], requires_grad=True)
b = Tensor([0., 0.], requires_grad=True)          

y = x @ W            
y = y + b            
y = y * 1.0           
y = y + Tensor([0.,0.])  
z = y.sum()           

print("\n===== GRAPH =====")
print_graph(z)

print("\n===== TOPO ORDER =====")
order = get_topo_order(z)
for i, node in enumerate(order):
    print(i, type(node._op).__name__ if node._op else "LEAF", node.shape)

print("\n===== AFTER OPTIMIZATION =====")
opt_order, replacement_map = remove_identity_ops(order)
for i, node in enumerate(opt_order):
    print(i, type(node._op).__name__ if node._op else "LEAF", node.shape)

for node in opt_order:
    if node._op is None:
        continue

    new_inputs = []
    for inp in node._inputs:
        if id(inp) in replacement_map:
            new_inputs.append(replacement_map[id(inp)])
        else:
            new_inputs.append(inp)

    node._inputs = new_inputs



print("\n===== EXECUTION =====")
result = execute(opt_order)
print("Result:", result.data)
print("real result: ", z.data)







