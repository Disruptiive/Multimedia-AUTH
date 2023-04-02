from collections import Counter
from itertools import chain

def make_tree(nodes):
    #while there are 2 or more symbols
    while len(nodes) > 1:
        #pick the 2 least probable ones
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        #create a new node that combines them, add it to the list and sort again
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]

#object to keep track of nodes and their children
class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right
    
    #traverse the tree, every edge to the left child is a trailing 0 every edge to the right child is a 1 
def huffman_encode_tree(node, code=''):
    #when it reaches a leaf it returns its encoded string 
    if type(node) is str: 
        return {node:code}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_encode_tree(l, code + '0'))
    d.update(huffman_encode_tree(r, code + '1'))
    return d

def huff(run_symbols):
    #count rle elements and calculate their probabilities
    elems = list(chain(*run_symbols))
    counted_elementes = dict(Counter(zip(elems[::2], elems[1::2])))
    p = sorted(counted_elementes.items(),key= lambda x:x[1],reverse=True)
    #convert every RLE entry to string [-4 4] -> '-4,4' 
    probabilities = []
    for i in range(len(p)):
        probabilities.append([",".join(map(str,p[i][0])),p[i][1]])

    #make tree based on their probabilities and encode every symbol
    tree = make_tree(probabilities)

    encoding = huffman_encode_tree(tree)
    #create the encoded string string 
    final_str = ''
    for i in run_symbols:
        final_str += encoding[",".join(map(str,i))]+' '
    return probabilities,final_str

def ihuff(frame_stream, frame_symbol_prob):
    run_symbols = []
    #make tree and encode it
    tree = make_tree(frame_symbol_prob)
    encoding = huffman_encode_tree(tree)
    #invert the dictionary and parse RLE rows as integers instead of strings eg '-1,0':010 -> 010:[-1,0] 
    inv_map = {v: [int(x) for x in k.split(",")] for k, v in encoding.items()}
    #decode it 
    for i in frame_stream.strip().split(" "):
        run_symbols.append(inv_map[i])
    return run_symbols