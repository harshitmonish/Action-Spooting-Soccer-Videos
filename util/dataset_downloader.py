import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

def downloadDataset(labels=True, 
    res_features=True, 
    baidu_features=True,
    path="drive/MyDrive/Action Spotting CSE 610/labels/"):

    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=path)

    if labels:
        mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], 
            split=["train","valid","test"])
        print("[INFO] downloaded labels")
        
    if res_features:
        mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", 
            "2_ResNET_TF2_PCA512.npy"], split=["train","valid","test","challenge"])
        print("[INFO] ResNet features")

    if baidu_features:
        mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", 
            "2_baidu_soccer_embeddings.npy"], split=["train","valid","test","challenge"])
        print("[INFO] Baidu features")
