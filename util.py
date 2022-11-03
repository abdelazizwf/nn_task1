# util.py

# Global variables to hold the species names
C1 = 'Adelie'
C2 = 'Gentoo'
C3 = 'Chinstrap'
CLASSES = [C1, C2, C3]

def transpose(l):
    """
    Transpose a 2D list and return it as a new list.
    """
    return [[row[i] for row in l] for i in range(len(l[0]))]

def extract_features(data, *features):
    """
    Extract features from 2D data by:
        1- transposing the data.
        2- selecting the rows where the first element is name
            of a desired feature (rows correspond to columns in
            the original data).
        3- re-transposing the selected data to return to its
            original shape.
    """
    new_data = []
    data_t = transpose(data)

    for column in data_t:
        if column[0] in features + ('species',):
            new_data.append(column)

    new_data = transpose(new_data)
    return list(new_data)

def get_class_slice(cls):
    """
    Return the slice object for a specific species by taking
    advantage of the fact that they are ordered.
    """
    slices = {
        C1: slice(1, 51),
        C2: slice(51, 101),
        C3: slice(101, 151)
    }
    return slices[cls]
