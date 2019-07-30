"""
Sampling script for finding the optimal pairs
"""
import numpy as np
DEBUG = True
if DEBUG:
    np.random.seed(7)

class Selector(object):
    def __init__(self, data, train_ratio=0.8, option='equality'):
        """
        Args:
            data: list of dictionaries each representing a clip
            option: options for how pairs are formed.
        """
        self.data = data
        self.option = option
        self.by_noun = self.organize_by_noun(self.data)
        self.train_split, self.val_split = self.training_val_split(train_ratio, self.by_noun)

        # option 1: fully connected. Returns all sample pairs. 
        # Better for when the number of clips is limited.

        # option 2: maximizing equality of the number of inter-class pairs 
        # and intra-class pairs for all classes. Step 1: calculate the minimum 
        # number of pairs within the different groups (self-selection included).
        # Step 2: sample max(threshold, minimum pairs) from all pairs of classes,
        # and from within classes.

        pass

    def training_val_split(self, train_ratio, by_noun):
        train = {} 
        val = {}
        for class_ in by_noun:
            train_section = by_noun[class_][:int(len(by_noun[class_]) * train_ratio)]
            val_section = by_noun[class_][int(len(by_noun[class_]) * train_ratio)+1:]

            train[class_] = train_section
            val[class_] = val_section

        return train, val # these are indices


    def organize_by_noun(self, data):
        by_noun = {}
        for idx, sample in enumerate(data):
            assert 'noun_class' in sample, '"noun_class" not in the sample.'
            if sample['noun_class'] not in by_noun:
                by_noun[sample['noun_class']] = [idx]
            else:
                by_noun[sample['noun_class']].append(idx)
        return by_noun

    def brute_force_list_all(self,indices, indices2=None):
        pairs = []
        if indices2 is None:
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    pairs.append((indices[i],indices[j]))
        else:
            for i in range(len(indices)):
                for j in range(len(indices2)):
                    pairs.append((indices[i], indices2[j]))
        return pairs

    def get_fully_connected_indices(self, data, sample_num=None):
        """
        Args:
            data: dictionary of clips per different class
            sample_num: either integer or decimal. if Integer, that will be interpreted as the maximum
                number of clips. If decimal, that will be interpreted as a percent of all fully connected
                pairs.
        """
        concat_indices = []
        for class_ in data:
            concat_indices.extend(data[class_])

        # fully connected on concat_data
        pairs = self.brute_force_list_all(concat_indices)

        if sample_num is not None:
            if type(sample_num) is float and sample_num > 1:
                pairs_indices = np.random.choice(range(len(pairs)), sample_num, replace=False).tolist()
            else:
                pairs_indices = np.random.choice(range(len(pairs)), int(len(pairs)*sample_num), replace=False).tolist()

            pairs = [pairs[idx] for idx in pairs_indices]

        return pairs

    def get_equitable_indices(self, data, min_threshold=20, max_threshold=500, bruteforcelist=True):
        
        # calculating the max and min number of pairs possible within classes and accross classes
        counts = {}
        for key_1 in data:
            for key_2 in data:
                if key_1 != key_2:
                    num_pairs = len(data[key_1]) * len(data[key_2]) # 
                else:
                    num_pairs = int(0.5*len(data[key_1])*(len(data[key_1]) - 1)) # n choose 2

                counts[(key_1, key_2)] = num_pairs

        required_pair_num = min(max(min_threshold, min(counts.values())), max_threshold)
        
        # selecting required
        pairs = []
        for class1 in data:
            for class2 in data:
                if class1 == class2:
                    if bruteforcelist:
                        all_choices = self.brute_force_list_all(data[class1])
                        num_choices = min(counts[(class1, class2)], required_pair_num)
                        choices = np.random.choice(list(range(len(all_choices))), num_choices, replace=False).tolist()
                        pairs.extend([all_choices[idx] for idx in choices])
                    else:
                        num_choices = min(counts[(class1, class2)], required_pair_num)
                        for i in range(num_choices):
                            # selecting two from the current list at random
                            choice = np.random.choice(data[class1], 2, replace=False).tolist()
                            pairs.append(tuple(choice))

                else:
                    if bruteforcelist:
                        num_choices = min(counts[(class1, class2)], required_pair_num)
                        all_choices = self.brute_force_list_all(data[class1], data[class2])
                        choices = np.random.choice(list(range(len(all_choices))), num_choices, replace=False).tolist()
                        pairs.extend([all_choices[idx] for idx in choices])
                    else:
                        num_choices = min(counts[(class1, class2)], required_pair_num)
                        for i in range(num_choices):
                            # selecting two from the current list at random
                            choice1 = np.random.choice(data[class1], 1, replace=False)[0]
                            choice2 = np.random.choice(data[class1], 1, replace=False)[0]
                            pairs.append((choice1, choice2))
        return pairs

    def get_indices(self, set_type, min_threshold=40, max_threshold=500, sample_num=0.8):

        assert set_type == 'train' or set_type == 'val', 'set_type has to be either train or val.'

        if self.option == 'fullyconnected':
            pairs = self.get_fully_connected_indices(self.train_split if set_type =='train' else self.val_split,
                                                            sample_num=sample_num)

        elif self.option == 'equality':
            pairs = self.get_equitable_indices(self.train_split if set_type == 'train' else self.val_split,
                                                        min_threshold=min_threshold, max_threshold=max_threshold)
        else:
            raise ValueError('self.option is not valid. Double check.')
        
        return pairs

if __name__ == '__main__':
    import pickle
    with open('rm_me.pkl', 'rb') as f:
        training_data = pickle.load(f)
    my_selector = Selector(training_data, option='equality', train_ratio=0.6)

    train_pair_indices = my_selector.get_indices('train')
    val_pair_indices = my_selector.get_indices('val')
    import ipdb; ipdb.set_trace()








