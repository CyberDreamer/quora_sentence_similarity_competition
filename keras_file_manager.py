from keras.models import model_from_json

def SaveToJSon(model, fileName):
    model_json = model.to_json()
    with open(fileName + ".net", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(fileName + ".wgt")
    print("model saved to disk")

def LoadFromJSon(fileName):
    json_file = open(fileName + ".net", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(fileName + ".wgt")

    print("Loaded model from disk")

    return loaded_model