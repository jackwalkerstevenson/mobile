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

layer_height = 1
linear_density = 1
parent_linkage = 0.5
leaf_linkage = 0
first_leaf_mass = 20


def edge_mass(horizontal, vertical):
    edge_length = sqrt(horizontal**2 + vertical**2)  # hypotenuse for now
    return linear_density * edge_length


class Node:
    gen_id = itertools.count()  # using this as coeff index
    # therefore have to create all the leaves first

    def __init__(self, left=None, right=None, dist=None):
        self.ID = next(Node.gen_id)
        self.left = left  # note no actual distinction bt left and right
        self.right = right
        self.dist = dist

    def is_leaf(self):
        return self.left is None and self.right is None

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
            edge = edge_mass(self.dist, layer_height)
            coeff_row[-1] += 2 * edge + parent_linkage
        return coeff_row

    def constraint(self):
        return self.left.sum() - self.right.sum()

    def __repr__(self):
        return (f'(Node {self.ID} with left {self.left} and right {self.right}'
                f' and distance {self.dist})')


""" test tree structure
                      a
                /                   \
                b                    c
            /     |            /       \
         A         d          e         f
                /     |      /  |      /   |
              B        C    D    E    F     G
"""

A = Node()
B = Node()
C = Node()
D = Node()
E = Node()
F = Node()
G = Node()
d = Node(B, C, 1)
e = Node(D, E, 5)
f = Node(F, G, 1)
b = Node(A, d, 3)
c = Node(e, f, 2)
a = Node(b, c, 0.5)

nonleaves = [a, b, c, d, e, f]
num_nonleaves = len(nonleaves)
num_leaves = num_nonleaves + 1

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
    raise ValueError("Input values yielded no solution. Consider increasing "
                     "placeholder mass for first leaf")
