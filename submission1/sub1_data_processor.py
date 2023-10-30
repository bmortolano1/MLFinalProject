import numpy as np

def get_train_data():

    # Return 2-D numpy array containing test data

    table = np.empty([0,0])

    with open('../data/train_final.csv', 'r') as f:
        for line in f:
            terms = np.char.split(line.strip(), ',').tolist()
            if np.size(table) == 0:
                table = terms
            else:
                table = np.vstack([table, terms])

    features = table[1:, 1:-1]
    labels = table[1:, -1]

    return features, labels

def get_test_data():

    # Return 2-D numpy array containing test data

    table = np.empty([0,0])

    with open('../data/test_final.csv', 'r') as f:
        for line in f:
            terms = np.char.split(line.strip(), ',').tolist()
            if np.size(table) == 0:
                table = terms
            else:
                table = np.vstack([table, terms])

    features = table[1:, :]

    return features
