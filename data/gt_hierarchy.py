import numpy as np
import os
import json
import random
import pickle
import pandas as pd

DEBUG = False
if DEBUG:
    random.seed(7)
    np.random.seed(7)
    force_feed = ['chicken', 'carrot']

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

def get_noun_class_frame_frequency(object_file_path):

    # loading the counts csv and making a dictionary
    obj_df = pd.read_csv(object_file_path)
    counts = {}
    for class_id in list(set(obj_df.noun_class)): # a number
        counts[class_id] = len(obj_df.loc[((obj_df['noun_class']==class_id) & (obj_df['bounding_boxes']!='[]'))])

    return counts

# TODO: add required_training_knowns file, which contains the classes which 
# must be in the training known section
def get_known_unknown_split(tree_file='hierarchyV1.json', 
                            required_training_knowns='EK_Imagenet_intersection.txt',
                            max_training_knowns = 8,
                            label_csv_path = '/vision/group/EPIC-KITCHENS/annotations/EPIC_train_object_labels.csv',
                            noun_key_path = '/vision/group/EPIC-KITCHENS/annotations/EPIC_noun_classes.csv',
                            select_random_classes = True):
    """ gets the training/testing known/unknown split
    """
    # training knowns and unknowns
    # testing unknowns
    training_unknowns = []
    testing_unknowns = []
    training_knowns = [] 
    if DEBUG:
        print('WARNING WE ARE FORCE FEEDING')
        max_training_knowns=len(force_feed)

    if required_training_knowns is not None:
        # loading the required training known classes
        with open(required_training_knowns, 'r') as f:
            lines = f.read()

        required_training_knowns = [element for element in lines.split('\n') if len(element)>0]
        if len(required_training_knowns) > max_training_knowns:
            # choose a subset of them
            # if random:
            if select_random_classes:
                include_training_knowns = np.random.choice(required_training_knowns, max_training_knowns, replace=False).tolist() if not DEBUG else force_feed

            else:
                counts = get_noun_class_frame_frequency(label_csv_path)
                # TODO: maximize frequency of the max_training_knowns
                class_key_df = pd.read_csv(noun_key_path)
                class_key_dict = dict(zip(class_key_df.class_key, class_key_df.noun_id))

                required_and_counts = []
                for class_ in required_training_knowns:
                    if class_key_dict[class_] not in counts:
                        print('WARNING: {} NOT DETECTED'.format(class_))
                        required_and_counts.append((class_, 0))
                    else:
                        required_and_counts.append((class_, counts[class_key_dict[class_]]))

                include_training_knowns = [element[0] for element in sorted(required_and_counts, key=lambda x: x[1], reverse=True)][:max_training_knowns]

            found = {element: False for element in include_training_knowns}
    
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
                # take out the ones that are reuqired training_knowns
                if layer3_class not in include_training_knowns and layer3_class not in required_training_knowns:
                    this_branch_candidates.append(layer3_class)
                else:
                    num_required_training_unknowns += 1
                    found[layer3_class] = True

            random.shuffle(this_branch_candidates)
            # splitting into three even groups, in the following pirority order:
            # training known, testing unknown, training unknown
        
            
            a = int(np.ceil(len(this_branch_candidates)/3.0))
            # a = int(np.ceil((len(this_branch_candidates) + num_required_training_unknowns)/3.0))
            # if num_required_training_unknowns > a:
            #     b = num_required_training_unknowns
            # else:
            #     b = a - num_required_training_unknowns
            
            # TODO: FIX WASTE!!!
            training_known = this_branch_candidates [:a]
            training_knowns.extend(training_known)
            training_unknown= this_branch_candidates [a:2*a]
            training_unknowns.extend(training_unknown)
            testing_unknown = this_branch_candidates [2*a:]
            testing_unknowns.extend(testing_unknown)
    
     
    training_knowns = include_training_knowns + np.random.choice(training_knowns, max_training_knowns - len(include_training_knowns), replace=False).tolist()

    assert all(list(found.values())), \
            '{} were not found in the hierarchy. Check spelling in original file.'.format([element for element in found if not found[element]])

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

    get_known_unknown_split()
