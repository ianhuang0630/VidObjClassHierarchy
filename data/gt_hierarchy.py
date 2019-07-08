import numpy as np
import os
import json 

def get_tree_distance(class1, class_2, tree_file='hierarchyV1.json'):
    """ gets the tree distance between two classes
    """
    assert os.path.exists(tree_file), '{} does not exist'.format(tree_file)
    
    with open(tree_file, 'r') as f:
        hierarchy = json.load(f)
    

    pass

def get_tree_position(class_, tree_file='hierarchyV1.json'):
    """ gets vector encoding of the position within the tree
    """
    pass

def get_random_known_unknown_split(tree_file='hierarchyV1.json'):
    """ gets the training/testing known/unknown splits
    """
    pass

if __name__=='__main__':
    pass 
