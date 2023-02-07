import graphviz
from copy import deepcopy
from IPython.display import Latex
import numpy as np
import math
import copy

def reshape_grad(grad, shape):
    if math.prod(grad.shape) == math.prod(shape):
        new_grad = grad.reshape(shape)
    elif math.prod(shape) == 1:
        new_grad = np.sum(grad).reshape(1, 1)
    elif grad.shape[0] == shape[0]:
        if grad.shape[1] > shape[1]:
            new_grad = np.sum(grad, axis=-1).reshape(-1, 1)
        else:
            raise ValueError(f"Cannot increase dimensions from {grad.shape} to {shape}")
    elif grad.shape[1] == shape[1]:
        if grad.shape[0] > shape[0]:
            new_grad = np.sum(grad, axis=0).reshape(1, -1)
        else:
            raise ValueError(f"Cannot increase dimensions from {grad.shape} to {shape}")
    else:
        raise ValueError("Cannot reshape gradient.")
    return new_grad

class Node():
    def __init__(self, *args, out=None, desc=None):
        """
        Initialize a node.
        :param args: Any arguments needed to compute the forward pass.
        :param out: A string representing the output of the node.
        :param desc: A string describing the node.
        """
        self.nodes = args
        self.desc = desc
        self.out = out
        self.needs_grad = False
        self.parent_nodes = []
        self.grad_cache = None
        if self.desc and not self.out:
            self.out = self.desc

    def visualize(self, graph, backward=False):
        """
        Generate a graphviz graph of the node.
        :param graph: a graphviz digraph object.
        :param backward: Whether to render forward or backward.
        """
        self_id = str(id(self))
        self_name = type(self).__name__
        if self.desc:
            self_name = self.desc
        graph.node(self_id, self_name)
        # Check for leaf node
        if self.nodes is None:
            return
        for node in self.nodes:
            node_id = str(id(node))
            node.visualize(graph, backward=backward)
            if backward:
                label = f"d({node.out})"
                # ensure that x and y don't get a grad label
                if isinstance(node, Parameter) and not node.needs_grad:
                    label = None
                graph.edge(self_id, node_id, label=label, color="red")
            else:
                graph.edge(node_id, self_id, label=node.out)

    def zero_grad(self):
        """
        Zero out the gradients on each parameter.
        """
        if self.needs_grad:
            self.grad = None
            self.derivative = []
        self.grad_cache = {}
        if self.nodes is None:
            return
        for node in self.nodes:
            node.zero_grad()

    def apply_fwd(self):
        # Leaf nodes just return
        if self.nodes is None:
            return self.forward()
        args = []
        # Loop through child nodes and call apply
        for node in self.nodes:
            if self not in node.parent_nodes:
                node.parent_nodes.append(self)
            args.append(node.apply_fwd())
        # Cache inputs for backprop
        self.cache = args
        return self.forward(*args)

    def apply_bwd(self, grad=None):
        # If it is a terminal node, then get gradient from input
        if len(self.parent_nodes) == 0:
            new_grad = self.backward(grad)
        else:
            in_grad = 0
            # Loop across parent nodes (which pass gradient down)
            for node in self.parent_nodes:
                self_id = id(self)
                if self_id not in node.grad_cache:
                    # This means not all parents have finished computing yet
                    return
                # Sum gradient coming from parent nodes
                in_grad += node.grad_cache[self_id]
            # Pass gradient across this node
            new_grad = self.backward(in_grad)
        if not isinstance(new_grad, tuple):
            new_grad = (new_grad,)
        # End chain if we hit a leaf node
        # Leaf nodes will set the self.grad property in backward
        if self.nodes is None:
            return None

        # Reshape gradients if necessary
        reshaped_grads = []
        for g, arg in zip(new_grad, self.cache):
            # Some parameters are single numbers
            has_shape = hasattr(g, "shape") and hasattr(arg, "shape")
            if has_shape and g.shape != arg.shape:
                g = reshape_grad(g, arg.shape)
            reshaped_grads.append(g)
        # Set grad cache for child nodes to get later
        for node, grad in zip(self.nodes, reshaped_grads):
            node_id = id(node)
            self.grad_cache[node_id] = grad
        # Call child nodes to calculate their gradients
        for node in self.nodes:
            node.apply_bwd()

    def generate_graph(self, backward=False):
        graph = graphviz.Digraph('fwd_pass', format="png", strict=True)
        graph.attr(rankdir='LR')
        # this recurses through nodes
        self.visualize(graph, backward=backward)
        return graph

    def generate_derivative_chains(self, chain=None):
        if chain is None:
            chain = []
        # Only set derivatives on leaf nodes
        if self.nodes is None:
            if self.needs_grad:
                self.derivative.append(chain)
            return
        for node in self.nodes:
            node_chain = deepcopy(chain)
            partial = f"\\frac{{\partial {self.out}}}{{\partial {node.out}}}"
            node_chain.append(partial)
            node.generate_derivative_chains(node_chain)

    def display_partial_derivative(self):
        flat_eqs = ["*".join(item) for item in self.derivative]
        lhs = f"\\frac{{\partial}}{{\partial {self.out}}}"
        rhs = " + \\\\".join(flat_eqs)
        return f"{lhs} = {rhs}"

    def forward(self, *args):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()

def display_chain(eq):
    return Latex(f"${eq}$")

class Parameter(Node):
    def __init__(self, data, desc=None, needs_grad=True):
        super().__init__(data, desc=desc)
        self.data = data
        self.nodes = None
        self.needs_grad = needs_grad

    def forward(self):
        return self.data

    def backward(self, grad):
        if not self.needs_grad:
            return
        grad = reshape_grad(grad, self.data.shape)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad