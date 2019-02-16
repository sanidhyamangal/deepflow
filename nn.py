"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from deepflow import *

w, x, y, z = Input(), Input(), Input(), Input()

f = Add([x, y, z, w])

feed_dict = {x: 4, y: 5, z: 10, w:1}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# output the sum of all the values 
print(" Sum  of all the nodes  {} (according to deepflow)".format(output))