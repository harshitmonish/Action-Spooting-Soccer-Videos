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

# Report
Please refer to Action_Spotting_Report.pdf for details.
