# util.py

C1 = 'Adelie'
C2 = 'Gentoo'
C3 = 'Chinstrap'
CLASSES = [C1, C2, C3]

def transpose(l):
    return [[row[i] for row in l] for i in range(len(l[0]))]

def extract_features(data, f1, f2):
    new_data = []
    data_t = transpose(data)

    for column in data_t:
        if column[0] in [f1, f2, 'species']:
            new_data.append(column)

    new_data = transpose(new_data)
    return list(new_data)

def get_class_slice(cls):
    slices = {
        'Adelie': slice(1, 51),
        'Gentoo': slice(51, 101),
        'Chinstrap': slice(101, 151)
    }
    return slices[cls]
