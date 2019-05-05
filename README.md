# CartPole-v1-DeepQ-AI

This is a simple implementation of deepQlearning Algorithm for the openai gym cartpole-v1

To run the model call the trainer class constructor and pass the number of episodes required for training.
For me about 150 episodes seems to work fine.
Then call thetrain method of the same object to start the training phase.
After training, the plots for evaluation would automatically generated.
To test the model call the play method. It takes how many times you want to play as a parameter.
Finally to save the trained model call the save_model method.

For the training phase i got the following evaluation plot --

![Training](https://github.com/siddharthkale97/CartPole-v1-DeepQ-AI/blob/master/High_training_average.png)

For the hundred evaluation plays --

![Testing](https://github.com/siddharthkale97/CartPole-v1-DeepQ-AI/blob/master/High_100_play_average.png)

As wee can see for both training and testing the average for 100 consecutive plays was well above 200.
