# Action Spotting in Soccer broadcast Untrimmed videos

In team sports like football both person level information and group level contexts are crucial for event recognition. The task of spotting consists of finding the anchor time that identifies
an event in a video and then classify the event to appropriate class. A common pipeline comprise of proposing temporal segments which are in turn further pruned and classified. Common
methods for activity classification and detection make use of dense trajectories, actions’ estimation, recurrent neural network, tubelets and handcrafted features. In order to recognize
or detect activities within a video, a common practice consists of aggregating local features and pooling them, looking for a consensus of characteristics. In this project we focus our
analysis on action spotting in soccer broadcast videos.

# Dataset
The SoccerNet-v2 dataset comprise broadcast videos of total 550 games of total 764 hours with 720p and 224p resolutions at 25fps. We split the data into train set comprising of 300 games,
test set of 100 games and validation set of 100 games. The challenge set comprise of 50 games. There are total of 110,458 annotated actions in the dataset on average of 221 actions per game, 1 action
every 25 seconds. Each label is further marked as visible and non-visible. We have used the features provided.

# Implementation
The action spotting problem is 2-fold, i.e. it consists the time of the action and the category/class of the action. The proposed framework first classifies the timestamps of the actions in the entire game and then it classifies each of the timestamp. The system architecture is shown in following figure: 

![Alt text](https://github.com/harshitmonish/Action-Spooting-Soccer-Videos/blob/main/img/architecture.png?raw=true "Model Architecture")


The predictions from the Naive Classifier are used to generate action windows. Considering the current frame as 0, t frames before and after the predicted frames are grouped together which forms the action window. t is set to 5 for this implementation. These windows are passed as an input to the level 1 Bi-GRU model. The level-1 Bi-GRU model consists 2 bidirectional GRU layers stacked on top of each other. The output of the model is a 17 node layer, giving the probability of an event using
softmax activation. Based on the confidence of the level-1 model, the second level classification is decided. If the level-1 classifier predicts the action with less probability, the window is then passed to the level-2 classifier to again perform classification on it. The level-2 classifier is same as the level-1 classifier
with stacked bidirectional GRU layers. The key differences between the level-1 and level-2 classifier is that the level-2 classifier is trained specifically on pre/post frames of the predicted time of the action and also, it is trained on sub-group classification. Since the frames before and after the action may capture different data, hence, they will help us to correctly classify these similar actions.

# Evaluation Metric
* The performance is assessed by the Average mAP metric
* * A predicted action spot is positive if it falls within the given tolerance x of a ground truth timestamp from the same class.
* The Average Precision (AP) based on PR curves is computed and then averaged over the classes( mAP ), after which the Average mAP is the AUC of the mAP at different tolerances x.

# Results:
For tight bound of 5sec our model gave mAP of 35.17 on test data and 36.71 on challenge set as shown in the Image:
![Alt text](https://github.com/harshitmonish/Action-Spooting-Soccer-Videos/blob/main/img/Results.JPG?raw=true "Experimental Results") 

# Conclusion: 
In this project we have implemented a multimodal architecture for action spotting task in soccer videos using Soccernet-v2 dataset.The experimental results shows that the proposed scheme can incorporate pre-event and post-event frames for classification task, especially for closely related events.

# Report
Please refer to Action_Spotting_Report.pdf and Action_Spotting_Presentation for details.
