import re
from Ottergrad.autograd import Tensor as Node


def str2ops(func: str):
    def add(node):
        npops = ["np.exp", "np.tan", "np.tanh", "np.maximum", "np.where", "np.var", "np.mean", "np.sqrt", "np.dot",
                 "np.maximum", "np.minimum", "np.sum"]
        primops = ["[+-]", "[*/@]{1}", "[*]{2}"]
        bidict = ["+", "-", "*", "/", "@"]

        if re.match("^[0-9\.]*$", node.gettype()) is not None:
            node.data = float(node.gettype())

        for prim in primops:
            pattern = str(prim).replace(", ", "")
            ops = re.search(pattern, node.gettype())

            if ops is not None:
                ops = ops.group(0)

                if node.gettype()[node.gettype().find(ops) + 1:] != "":
                    node.left = Node(None, node.gettype()[node.gettype().find(ops) + 1:])
                    # if re.match("^[0-9\.]*$", node.gettype()[node.gettype().find(ops) + 1:]) is not None :
                    #     node.getleft().data = float(node.gettype()[node.gettype().find(ops) + 1:])
                    add(node.getleft())

                if ops in bidict and node.gettype()[:node.gettype().find(ops)] != "":
                    node.right = Node(None, node.gettype()[:node.gettype().find(ops)])

                    # if re.match("^[0-9\.]*$", node.gettype()[:node.gettype().find(ops)]) is not None :
                    #     node.getright().data = float(node.gettype()[:node.gettype().find(ops)])
                    add(node.getright())

                node.type = ops
                return
        return

    func = func.replace(" ", "")

    root = Node(None, func)
    add(root)


str2ops("1 + 3 * 4 * a")
