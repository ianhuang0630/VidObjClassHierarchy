import numpy as np
import os
import json 

def tree_dist(loc1, loc2):
    lock = False
    dist = 0
    for i in range(len(loc1)):
        if lock:
            dist += 2
        if loc1[i] != loc2[i] and not lock:
            lock = True
            dist += 2
    return dist        

# NOTE: below only applies for 3 layer hierarchy
def get_tree_distance(class1, class2, tree_file='hierarchyV1.json'):
    """ gets the tree distance between two classes
    """
    assert os.path.exists(tree_file), '{} does not exist'.format(tree_file)
    
    with open(tree_file, 'r') as f:
        hierarchy = json.load(f)
    assert type(hierarchy) == dict
    
    locations_per_layer3_class = {}
    for layer1_class in hierarchy:
        for layer2_subhierarchy in hierarchy[layer1_class]:
            layer2_class = list(layer2_subhierarchy.keys())[0]
            subsubhierarchy = layer2_subhierarchy[layer2_class]
            # subsubhierarchy should be a list 
            for layer3_class in subsubhierarchy:
                locations_per_layer3_class[layer3_class] = (layer1_class, layer2_class)
    
    if type(class1)!=list:
        loc1 = locations_per_layer3_class[class1]
        loc2 = locations_per_layer3_class[class2]
        return tree_dist(loc1, loc2)          
    else:
        assert len(class1) == len(class2)
        distances = []
        for i in range(len(class1)):
            loc1 = locations_per_layer3_class[class1[i]]
            loc2 = locations_per_layer3_class[class2[i]]
            distances.append(tree_dist(loc1, loc2))
        return distances  

def get_tree_position(class_, known_classes, tree_file='hierarchyV1.json'):
    """ gets vector encoding of the position within the tree
    """
    assert os.path.exists(tree_file), '{} does not exist'.format(tree_file)
    
    with open(tree_file, 'r') as f:
        hierarchy = json.load(f)
    assert type(hierarchy) == dict
    
    locations_per_layer3_class = {}
    known_classes_set = set(known_classes)

    encoding = None
    counter1 = 0
    for layer1_class in hierarchy:
        counter2 = 0
        for layer2_subhierarchy in hierarchy[layer1_class]:
            layer2_class = list(layer2_subhierarchy.keys())[0]
            subsubhierarchy = layer2_subhierarchy[layer2_class]
            # subsubhierarchy should be a list 
            counter3 = 0
            for layer3_class in subsubhierarchy:
                if layer3_class == class_ and layer3_class in known_classes_set:
                    encoding = np.array([counter1, counter2, counter3])
                    return encoding
                    counter3 += 1
                elif layer3_class in known_classes_set:
                    counter3 += 1
            counter2 += 1
        counter1 += 1

    raise ValueError('{} is not an existing class in current hierarchy') 

def get_random_known_unknown_split(tree_file='hierarchyV1.json'):
    """ gets the training/testing known/unknown splits
    """
    pass

if __name__=='__main__':
    print('distance between ONION and PEACH is {}'.\
            format(get_tree_distance('onion', 'peach')))

    print('distance between ONION and POTATO is {}'.\
            format(get_tree_distance('onion', 'potato')))

    print('distance between ONION and BOX is {}'.\
            format(get_tree_distance('onion', 'box')))

    print(get_tree_position('oil', ['oil']))
    
