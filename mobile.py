"""Calculate masses of leaves to balance the branches of a mobile sculpture

Arguments:
filename -- path/name of input file
    input file format tbd

This script is for designing a Calder-style mobile to represent a dendrogram.
Assumptions about the desired sculpture:
-Binary tree (exactly 2 children per node, or 0 children = leaf)
-Each pair of edges spans symmetrical horizontal distances
-But different pairs of branch edges can have different horizontal distances
-Layers are separated by the same vertical distance
-Linkages from a branch to a node have a constant mass m (e.g. fishing swivel)
-Linkages from a leaf edge to its pendant mass have a constant mass
-Wire-like edge material has constant linear density

"""
from math import sqrt
import itertools
import numpy as np

layer_height = 4
linear_density = 1
parent_linkage = 0.5
leaf_linkage = 0
first_leaf_mass = 100

num_leaves = 3
num_nonleaves = num_leaves - 1


def edge_mass(horizontal, vertical):
    edge_length = sqrt(horizontal**2 + vertical**2)  # hypotenuse for now
    return linear_density * edge_length


class Node:
    gen_id = itertools.count()  # using this as coeff index
    # therefore have to create all the leaves first

    def __init__(self, left=None, right=None, horiz=None):
        self.ID = next(Node.gen_id)
        self.left = left  # note no actual distinction bt left and right
        self.right = right
        self.horiz = horiz

    def is_leaf(self):
        return self.left is None and self.right is None

    def set_horiz(self, horiz):
        self.horiz = horiz

    def set_left(self, child):
        if self.horiz is None:
            raise ValueError("attempted to add child without setting horiz")
        self.left = child

    def set_right(self, child):
        if self.horiz is None:
            raise ValueError("attempted to add child without setting horiz")
        self.right = child

    def sum(self):
        if self.is_leaf():
            # if leaf, mass coefficient array is just 1 for your own ID
            coeff_row = np.zeros(num_leaves + 1)  # last col is ordinate
            coeff_row[self.ID] = 1
        else:
            # if parent, mass coefficient array is sum of child arrays
            left = self.left.sum()
            right = self.right.sum()
            coeff_row = left + right
            # plus linkage mass for yourself and 2 edge weights in ordinate col
            edge = edge_mass(self.horiz, layer_height)
            coeff_row[-1] += 2 * edge + parent_linkage
            print(f'adding to coeff_row: {2 * edge + parent_linkage}')
        return coeff_row

    def constraint(self):
        return self.left.sum() - self.right.sum()

    def __repr__(self):
        return (f'(Node {self.ID} with left {self.left} and right {self.right}'
                f' and distance {self.horiz})')


""" test tree structure
                    a
                /       \
                A       b
                    /       \
                    B       C
"""

A = Node()
B = Node()
C = Node()
b = Node(B, C, 3)
a = Node(A, b, 10)
nonleaves = [b, a]
temp_coeff_rows = num_leaves
temp_coeff_cols = num_leaves + 1
temp_coeffs = np.zeros((temp_coeff_rows, temp_coeff_cols))
# set first leaf mass to make numpy happy
temp_coeffs[-1, 0] = 1
temp_coeffs[-1, -1] = -1 * first_leaf_mass  # bc ordinate sign will be flipped
for i, node in enumerate(nonleaves):
    temp_coeffs[i] = node.constraint()
coeffs = temp_coeffs[:, 0:-1]
ords = -1 * temp_coeffs[:, -1]  # move ordinates to other side: flip sign
solution = np.linalg.solve(coeffs, ords)
print(f'solution:\n{solution}')
if not all(i > 0 for i in solution):
    raise ValueError("Input values yielded no solution. Consider increasing placeholder mass for first leaf")
