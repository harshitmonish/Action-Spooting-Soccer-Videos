# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 03:36:12 2022

@author: harsh
"""

from sklearn.metrics import precision_score

context = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1., 
                    0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5], dtype="float")
context = np.expand_dims(context, axis=1)


label2event_2 = {
    0: "Goal",
    1: "Shots on target",
    2: "Shots off target",
    3: "Corner",
}

label2event_4 = {
    0: "Yellow card",
    1: "Red card",
    2: "Yellow->red card",
    3: "Offside",
}

label2event_5 = {
    0: "Clearance",
    1: "Ball out of play",
    2: "Throw-in",
}

label2event_6 = {
    0: "Foul",
    1: "Indirect free-kick",
    2: "Direct free-kick",
}


map = {
    0: [1, 0, 0],
    -1: [0, 1, 0],
    1: [0, 0, 1],
}

def contextLabel(labels):
    # labels = labels * context
    label = np.sum(labels, axis=0)
    return label

def binaryLabel(labels):
    # label = np.sum(labels, axis=0)
    # val = np.sum(labels)
    # return map[np.sum(labels)]
    if np.sum(labels) == 0:
        return 0
    else:
        return 1

def divideGame(features, labels, binary=True, stride=1):
    feat_ = []
    lab_ = []

    for i in range(0, len(features), stride):
        if i+timestep > len(features):
            break
        feat_.append(features[i : i+timestep])
        if binary:
            lab_.append(binaryLabel(labels[i : i+timestep]))
        else:
            lab_.append(contextLabel(labels[i : i+timestep]))

    feat_ = np.asanyarray(feat_)
    lab_ = np.asanyarray(lab_)
    feat_ = np.reshape(feat_, (feat_.shape[0], feat_.shape[-1]))
    return (feat_, lab_)

def getBinaryClassWeights(labels):
    n_samples = len(labels)
    n_classes = 3

    class_count = []
    class_count.append(len(np.where(labels == 0)[0]))
    class_count.append(len(np.where(labels < 0)[0]))
    class_count.append(len(np.where(labels > 0)[0]))

    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights))
    return dict(zip(class_labels, class_weights))


# get windows from predictions
def getWindows(pred, threshold=0.2):
    windows = []
    pred = pred > threshold
    idx = 0

    while idx < len(pred):
        if pred[idx]:
            start = idx
            while idx < len(pred) and pred[idx]:
                idx += 1
            windows.append([start, idx-1])
        idx += 1
    
    return np.asarray(windows)

# segment actions from game for event classification
def segmentActions(windows, features, labels, max_window=15):
    feat_ = []
    lab_ = []

    for (start, end) in windows:
        temp_feat = []
        temp_feat.extend([np.array(features[f]) for f in range(start, end+1)])
        if end+1 - start < max_window: 
            temp_feat.extend(np.array([0] * features.shape[-1]) for _ in range(max_window - (end+1 - start)))

        temp_feat = np.asarray(temp_feat)
        feat_.append(np.array(temp_feat))

        temp_lab = contextLabel(labels[start : end+1])
        if np.sum(temp_lab) > 1:
            temp_lab = temp_lab / np.sum(temp_lab)
        lab_.append(temp_lab)

    feat_ = np.array(feat_)
    lab_ = np.array(lab_)
    return (feat_, lab_)

def getEventWindows(features, labels):
    feat_ = []
    lab_ = []

    for (idx, label) in enumerate(labels):
        if np.sum(label) > 0:
            start = idx - 5
            end = idx + 5
            temp_feat = []
            if start < 0:
                arr = [np.array(features[f]) for f in range(0, 11)]
                assert(len(arr) == 11)
            elif end > len(labels)-1:
                arr = [np.array(features[f]) for f in range(len(labels)-11, len(labels))]
                assert(len(arr) == 11)
            else:
                arr = [np.array(features[f]) for f in range(start, end+1)]
                assert(len(arr) == 11)
            temp_feat.extend(arr)
            # if end+1 - start < max_window: 
            #     temp_feat.extend(np.array([0] * features.shape[-1]) for _ in range(max_window - (end+1 - start)))
            temp_feat = np.asarray(temp_feat)
            feat_.append(np.array(temp_feat))

            # temp_lab = contextLabel(labels[start : end+1])
            # if np.sum(temp_lab) > 1:
            #     temp_lab = temp_lab / np.sum(temp_lab)
            lab_.append(label)

    feat_ = np.array(feat_)
    lab_ = np.array(lab_)
    

def getClassWeights(labels):
    n_samples = len(labels)
    n_classes = len(labels[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in labels:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1

    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights))
    return dict(zip(class_labels, class_weights))


def accuracy_(pred, labels):
    total, acc = 0, 0
    for p, l in zip(pred, labels):
        if np.sum(p) == np.sum(l) and np.argmax(p) == np.argmax(l):
            acc += 1
        total += 1
    precisions = precision_score(labels, pred, average=None)
    print(f"mAP = {np.sum(precisions) / 7}")
    
    return (acc / total)


def sec2time(sec):
    min = sec // 60
    sec -= (min*60)
    if sec < 10:
        return str(min) + ":0" + str(sec)
    return str(min) + ":" + str(sec)

# save predictions
def savePreds(preds, windows, filepath):
    # labels = []
    predictions = {"annotations": []}
    half = 0
    for (pred_half, window_half) in zip(preds, windows):
        half += 1
        for (pred, window) in zip(pred_half, window_half):
            event = {}
            # get timestamp
            sec = window[0] + ((window[1] - window[0]) // 2)
            time = sec2time(sec)

            # get class
            action = GV.INVERSE_EVENT_DICTIONARY_V2[np.argmax(pred)]
            
            # labels.append([action, time])
            predictions["annotations"].append({"gameTime": str(half) + " - " + time,
                                            "label": action,
                                            "position": str(sec * 1000),
                                            "half": str(half),
                                            "confidence": str(np.max(pred))})
    
    res_path = "results/" + filepath[7:]
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(res_path + "results_spotting.json", "w") as f:
        f.write(json.dumps({"UrlLocal": filepath[7:-1],
            "predictions": predictions["annotations"]}, indent=4))
        
        
def getPrePostFrames(peak, features):
    pre, post = [], []
    start = peak - 10
    end = peak - 1
    try:
        arr = [np.array(features[f]) for f in range(start, end+1)]
        pre.extend(arr)
    except:
        arr = [np.array(features[f]) for f in range(0, 11)]
        pre.extend(arr)
    
    start = peak + 1
    end = peak + 10
    try:
        arr = [np.array(features[f]) for f in range(start, end+1)]
        post.extend(arr)
    except:
        arr = [np.array(features[f]) for f in range(len(features)-10, len(features))]
        post.extend(arr)

    return (pre, post)

def getSubclass(peak, features, label):
    pre, post = getPrePostFrames(peak, features)
    pre = np.expand_dims(pre, axis=0)
    post = np.expand_dims(post, axis=0)
    if label == 2:
        pre_pred = pre_2_model.predict((pre))
        # post_pred = post_2_model.predict(post)
        # if np.max(pre_pred) > np.max(post_pred):
        return label2event_2[np.argmax(pre_pred)], pre_pred
        # return label2event_2[np.argmax(post_pred)], post_pred
    elif label == 4:
        # pre_pred = pre_4_model.predict(pre)
        post_pred = post_4_model.predict(post)
        # if np.max(pre_pred) > np.max(post_pred):
        #     return label2event_4[np.argmax(pre_pred)], pre_pred
        return label2event_4[np.argmax(post_pred)], post_pred
    elif label == 5:
        pre_pred = pre_5_model.predict(pre)
        # post_pred = post_5_model.predict(post)
        # if np.max(pre_pred) > np.max(post_pred):
        return label2event_5[np.argmax(pre_pred)], pre_pred
        # return label2event_5[np.argmax(post_pred)], post_pred
    else:
        pre_pred = pre_6_model.predict(pre)
        # post_pred = post_6_model.predict(post)
        # if np.max(pre_pred) > np.max(post_pred):
        return label2event_6[np.argmax(pre_pred)], pre_pred
        # return label2event_6[np.argmax(post_pred)], post_pred
        


def save_dataset():
    features, labels = [], []
    i = 1

    for idx in range(0, len(trainList), 50):
        features, labels = [], []

        for fp in range(idx, idx+50):
            # print(f"[INFO] storing games {idx+1}-{idx+51}")    
            
            filepath = "labels/" + trainList[fp] + "/"
            print(f"[INFO] {fp+1}/500 games processed")
            
            [features_1, features_2, labels_1, labels_2] = getGameData(filepath, 
                                                                       num_classes=17, 
                                                                       event_mapping=GV.EVENT_DICTIONARY_V2)
            if features_1 is None:
                continue
            
            features_1, _ = divideGame(features_1, labels_1)
            features_2, _ = divideGame(features_2, labels_2)
            
            pred_1 = model.predict_on_batch(features_1)
            pred_2 = model.predict_on_batch(features_2)
            
            windows_1 = getWindows(pred_1)
                windows_2 = getWindows(pred_2)

            features_1, labels_1 = divideGame(features_1, labels_1, binary=False)
            features_2, labels_2 = divideGame(features_2, labels_2, binary=False)

            [features_1, labels_1] = segmentActions(windows_1, features_1, labels_1)
            [features_2, labels_2] = segmentActions(windows_2, features_2, labels_2)
            
            if features_1 is None or len(features_1.shape) != 3 or \
                features_2 is None or len(features_2.shape) != 3:
                continue

            if len(features) != 0:
                features = np.concatenate((features, features_1, features_2), axis=0)
                labels = np.concatenate((labels, labels_1, labels_2), axis=0)
            else:
                features = np.concatenate((features_1, features_2), axis=0)
                labels = np.concatenate((labels_1, labels_2), axis=0)

        with open("dataset/train_" + str(i) +"_features.npy", "wb") as f:
            np.save(f, features)

        with open("dataset/train_" + str(i) + "_labels.npy", "wb") as f:
            np.save(f, labels)
    
        i += 1
        
