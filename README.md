# Machine-Learning

Helo guys! 
FINALLY!!! I succeed to make a simple recommendation system using collaborative filtering! 
But I think I still use the old version of Tensorflow. We can work on it later. 

I used a different dataset. A dataset that consists of userID, placeID, and ratings (I only used the total_rating).
If you want to try by yourself, it's easy:
1. Download the ipnyb file and data folder.
2. Run the ipnyb file using Jupyter Notebook.
3. Run all of the program sequentially.

Also, I added the other method, the popularity-based model. How to run it? same as above.
Also, there is the one using Tensorflow Recommenders. How to run it? same as above.

For further information, ask me by WA. 


TODO:
 - [x] Create the autoencoder collaborative filtering tf model
 - [x] Create the data preprocessing functions
 - [x] Create a training script
 - [ ] Test the scripts
 - [ ] Train the model
 - [ ] Create a deployment script (in tfjs or tfx or pure python)
 - [ ] Add the genres filtering


 ## Model Change Log
 - converted the model to use dense layers instead of manualy doing matrix multiplication
 - added MRMSE loss function, refer [here](https://arxiv.org/pdf/1708.01715.pdf)
