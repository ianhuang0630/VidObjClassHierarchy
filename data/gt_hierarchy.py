import numpy as np
import os
import json
import random

DEBUG = True
if DEBUG:
    random.seed(7)
    np.random.seed(7)

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


def survey_tree(known_classes, tree_file='hierarchyV1.json'):
    assert os.path.exists(tree_file), '{} does not exist'.format(tree_file)

    with open(tree_file, 'r') as f:
        hierarchy = json.load(f)
    assert type(hierarchy) == dict

    known_classes_set = set(known_classes)

    map_ = {}

    counter1 = 0
    for layer1_class in hierarchy:
        counter2 = 0
        for layer2_subhierarchy in hierarchy[layer1_class]:
            layer2_class = list(layer2_subhierarchy.keys())[0]
            subsubhierarchy = layer2_subhierarchy[layer2_class]
            # subsubhierarchy should be a list
            counter3 = 0
            for layer3_class in subsubhierarchy:
                if layer3_class in known_classes_set:
                    counter3 += 1
            map_[(counter1, counter2)] = (counter3, (layer1_class, layer2_class)) # number for the unknown class in this category
            counter2 += 1
        counter1 += 1
    return map_

def get_tree_position(class_, known_classes, tree_file='hierarchyV1.json'):
    """ gets vector encoding of the position within the tree
    """
    assert os.path.exists(tree_file), '{} does not exist'.format(tree_file)

    with open(tree_file, 'r') as f:
        hierarchy = json.load(f)
    assert type(hierarchy) == dict

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
                    counter3 += 1
                elif layer3_class in known_classes_set:
                    counter3 += 1
            counter2 += 1
        counter1 += 1

    return encoding

# TODO: add required_training_knowns file, which contains the classes which 
# must be in the training known section
def get_known_unknown_split(tree_file='hierarchyV1.json', 
                            required_training_knowns='EK_imagenet_intersection.txt',
                            max_training_unknowns = 8):
    """ gets the training/testing known/unknown split
    """
    # training knowns and unknowns
    # testing unknowns
    training_unknowns = []
    testing_unknowns = []
    training_knowns = []

    if required_training_knowns is not None:
        # loading the required training known classes
        with open(required_training_knowns, 'r') as f:
            lines = f.read()

        required_training_knowns = [element for element in lines.split('\n') if len(element)>0]
        if len(required_training_knowns) > max_training_unknowns:
            # choose a subset of them
            include_training_unknowns = list(np.random.choice(required_training_knowns, max_training_unknowns, replace=False)[0])
            found = {element: False for element in include_training_unknowns}
            training_unknowns.extend(include_training_unknowns)

    assert os.path.exists(tree_file), '{} does not exist'.format(tree_file)
    with open(tree_file, 'r') as f:
        hierarchy = json.load(f)
    assert type(hierarchy) == dict
    
    for layer1_class in hierarchy:
        for layer2_subhierarchy in hierarchy[layer1_class]:
            layer2_class = list(layer2_subhierarchy.keys())[0]
            subsubhierarchy = layer2_subhierarchy[layer2_class]
            this_branch_candidates = []
            num_required_training_unknowns = 0
            for layer3_class in subsubhierarchy:
                # take out the ones that are reuqired training_unknowns
                if layer3_class not in include_training_unknowns:
                    this_branch_candidates.append(layer3_class)
                else:
                    num_required_training_unknowns += 1
                    found[layer3_class] = True

            random.shuffle(this_branch_candidates)
            # splitting into three even groups, in the following pirority order:
            # training known, testing unknown, training unknown

            a = int(np.ceil((len(this_branch_candidates) + num_required_training_unknowns)/3.0))
            if num_required_training_unknowns > a:
                b = num_required_training_unknowns
            else:
                b = a - num_required_training_unknowns

            training_unknown = this_branch_candidates [:b]
            training_unknowns.extend(training_unknown)
            testing_known = this_branch_candidates [b:2*a]
            testing_knowns.extend(testing_known)
            training_unknown = this_branch_candidates [2*a:]
            training_unknowns.extend(training_unknown)

    assert all(list(found.values())), 
        'Some of the required training unknowns were not found in the hierarchy. Check spelling in original file.'

    return {'training_unknown': training_unknowns,
            'training_known': training_knowns,
            'testing_unknown': testing_unknowns}

if __name__=='__main__':
    print('distance between ONION and PEACH is {}'.\
            format(get_tree_distance('onion', 'peach')))

    print('distance between ONION and POTATO is {}'.\
            format(get_tree_distance('onion', 'potato')))

    print('distance between ONION and BOX is {}'.\
            format(get_tree_distance('onion', 'box')))

    print(get_tree_position('oil', ['oil']))
