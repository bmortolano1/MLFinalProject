import numpy as np

# Create dictionary to map string inputs to integers
string_to_int_dict = {
    # Work Class
'Private':1, 'Self-emp-not-inc':2, 'Self-emp-inc':3, 'Federal-gov':4, 'Local-gov':5, 'State-gov':6, 'Without-pay':7, 'Never-worked':8,

    # Education
'Bachelors':1, 'Some-college':2, '11th':3, 'HS-grad':4, 'Prof-school':5, 'Assoc-acdm':6, 'Assoc-voc':7, '9th':8, '7th-8th':9,
    '12th':10, 'Masters':11, '1st-4th':12, '10th':13, 'Doctorate':14, '5th-6th':15, 'Preschool':16,

    # Martial Status
'Married-civ-spouse':1, 'Divorced':2, 'Never-married':3, 'Separated':4, 'Widowed':5, 'Married-spouse-absent':6, 'Married-AF-spouse':7,

    # Occupation
'Tech-support':1, 'Craft-repair':2, 'Other-service':3, 'Sales':4, 'Exec-managerial':5, 'Prof-specialty':6, 'Handlers-cleaners':7,
    'Machine-op-inspct':8, 'Adm-clerical':9, 'Farming-fishing':10, 'Transport-moving':11, 'Priv-house-serv':12, 'Protective-serv':13, 'Armed-Forces':14,

    # Relationship
'Wife':1, 'Own-child':2, 'Husband':3, 'Not-in-family':4, 'Other-relative':5, 'Unmarried':6,

    # Race
'White':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 'Other':4, 'Black':5,

    # Sex
'Female':1, 'Male':2,

    # Native Country
'United-States':1, 'Cambodia':2, 'England':3, 'Puerto-Rico':4, 'Canada':5, 'Germany':6, 'Outlying-US(Guam-USVI-etc)':7, 'India':8,
    'Japan':9, 'Greece':10, 'South':11, 'China':12, 'Cuba':13, 'Iran':14, 'Honduras':15, 'Philippines':16, 'Italy':17, 'Poland':18,
    'Jamaica':19, 'Vietnam':20, 'Mexico':21, 'Portugal':22, 'Ireland':23, 'France':24, 'Dominican-Republic':25, 'Laos':26, 'Ecuador':27,
    'Taiwan':28, 'Haiti':29, 'Columbia':30, 'Hungary':31, 'Guatemala':32, 'Nicaragua':33, 'Scotland':34, 'Thailand':35, 'Yugoslavia':36,
    'El-Salvador':37, 'Trinadad&Tobago':38, 'Peru':39, 'Hong':40, 'Holand-Netherlands':41,

    # Unknowns
'?':-1
}
def get_train_data():

    # Return 2-D numpy array containing test data

    table = np.empty([0,0])

    with open('../data/train_final.csv', 'r') as f:
        lines = f.readlines()[1:]
        f.close()
        for line in lines:
            terms = np.char.split(line.strip(), ',').tolist()

            # Convert terms from strings to numeric values
            for i in range(np.size(terms)):
                term = terms[i]
                if term.isnumeric():
                    term = float(term)
                else:
                    term = string_to_int_dict[term]
                terms[i] = term

            if np.size(table) == 0:
                table = terms
            else:
                table = np.vstack([table, terms])

    table = handle_unknowns(table)

    features = table[:, :-1]
    labels = table[:, -1]

    return features, labels

def get_test_data():

    # Return 2-D numpy array containing test data

    table = np.empty([0,0])

    with open('../data/test_final.csv', 'r') as f:
        lines = f.readlines()[1:]
        f.close()
        for line in lines:
            terms = np.char.split(line.strip(), ',').tolist()

            # Convert terms from strings to numeric values
            for i in range(np.size(terms)):
                term = terms[i]
                if term.isnumeric():
                    term = float(term)
                else:
                    term = string_to_int_dict[term]
                terms[i] = term

            if np.size(table) == 0:
                table = terms
            else:
                table = np.vstack([table, terms])

    table = handle_unknowns(table)

    features = table[:, 1:] # Skip ID

    return features

def handle_unknowns(feature_mat):
    for i in range(np.size(feature_mat,0)):
        row = feature_mat[i]
        for j in range(np.size(row)):
            if row[j] == -1:
                feature_mat[i,j] = most_common_value(feature_mat[:, j])

    return feature_mat

def most_common_value(n):
    occurences = np.zeros(np.size(n, 0))
    values = np.unique(n)

    i_max = 0
    occ_max = 0

    for i in range(np.size(values, 0)):
        occurences[i] = np.count_nonzero(n == values[i])

        if occurences[i] > occ_max:
            occ_max = occurences[i]
            i_max = i

    return values[i_max]

def output_test_file(labels, file_name):
    with open(file_name, 'w') as f:
        f.write('ID,Prediction\n')
        for i in range(np.size(labels, 0)):
            f.write(str(i+1) + ',' + str(labels[i]) + '\n')
        f.close()