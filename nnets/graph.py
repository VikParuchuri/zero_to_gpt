import graphviz
from copy import deepcopy
from IPython.display import Latex
import numpy as np

def reshape_grad(orig, grad):
    if grad.shape != orig.shape:
        try:
            summed = np.sum(grad, axis=-1)
            if summed.shape != orig.shape:
                summed = summed.reshape(orig.shape)
        except ValueError:
            summed = np.sum(grad, axis=0)
        if summed.shape != orig.shape:
            summed = summed.reshape(orig.shape)
        return summed
    return grad

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
        if self.nodes is None:
            return
        for node in self.nodes:
            node.zero_grad()

    def apply_fwd(self):
        if self.nodes is None:
            return self.forward()
        args = []
        for node in self.nodes:
            args.append(node.apply_fwd())
        self.cache = args
        return self.forward(*args)

    def apply_bwd(self, grad):
        grads = self.backward(grad)
        if not isinstance(grads, tuple):
            grads = (grads,)
        if self.nodes is None:
            return None
        args = []
        for node, grad in zip(self.nodes, grads):
            args.append(node.apply_bwd(grad))

    def generate_graph(self, backward=False):
        graph = graphviz.Digraph('fwd_pass', format="png", strict=True)
        graph.attr(rankdir='LR')
        self.visualize(graph, backward=backward)
        return graph

    def generate_derivative_chains(self, chain=None):
        if chain is None:
            chain = []
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
        lhs = f"\\frac{{\partial L}}{{\partial {self.out}}}"
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
        grad = reshape_grad(self.data, grad)
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad