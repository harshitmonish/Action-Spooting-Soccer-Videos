import numpy as np
import glob
import json
import src.global_variables as GV

def getPath(filepath):
    return filepath[:-29]

def getGame(filepath):
    return filepath.split("/")[-1][0]

def listGames(dir, filename="games.txt"):

    games = set()

    for filepath in glob.glob(dir + "*/*/*/*.npy"):
        games.add(getPath(filepath))

    with open(filename, 'w') as f:
        for game in games:
            f.write(game.replace("\\", "/"))
            f.write("\n")

    print("[INFO] games file generated.")

def label2vector(folder_path, num_classes=17, 
    vector_size_1=2700, vector_size_2=2700,
    event_mapping=GV.EVENT_DICTIONARY_V2):

    label_path = folder_path + "Labels-v2.json"

    # Load labels
    labels = json.load(open(label_path))

    label_half1 = np.zeros((vector_size_1, num_classes))
    label_half2 = np.zeros((vector_size_2, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = seconds + 60 * minutes

        if event not in event_mapping:
            continue
        label = event_mapping[event]

        value = 1
        if annotation["visibility"] == "not shown":
            value = -1

        if half == 1:
            label_half1[frame][label] = value

        if half == 2:
            label_half2[frame][label] = value

    return label_half1, label_half2

def getGameData(gamepath, num_classes=17, event_mapping=GV.EVENT_DICTIONARY_V2):
    label_file = gamepath + "Labels-v2.json"
    feature_1_file = gamepath + "1_baidu_soccer_embeddings.npy"
    feature_2_file = gamepath + "2_baidu_soccer_embeddings.npy"

    try:
        # load data
        feature_1 = np.load(feature_1_file)
        feature_2 = np.load(feature_2_file)
        label_half1, label_half2 = label2vector(gamepath, 
            vector_size_1=feature_1.shape[0],
            vector_size_2=feature_2.shape[0],
            num_classes=num_classes, 
            event_mapping=event_mapping)

        # append both halves
        # features = np.append(feature_1, feature_2, axis=0)
        # labels = np.append(label_half1, label_half2, axis=0)

        # return [features, labels]

        return [feature_1, feature_2, label_half1, label_half2]
    except:
        # challenge data
        return [None] * 4
