import pickle

def SaveToObject(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def LoadFromObject(filename):
    f = open(filename, 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    return loaded_obj