# Machine-Learning

Helo guys! 
FINALLY!!! we succeed to make a recommendation system using collaborative filtering!  

The dataset (final_ratings_new_ver.csv) consists of userID, placeID, and ratings.
If you want to try by yourself, it's easy:
1. Download the ipynb file (New Ver.ipynb) and the data or just simply download all from Main Dataset folder.
2. Run the ipnyb file using Jupyter Notebook.
3. Run all of the program sequentially.

To deploy the model : we can use server.py file, requirements.txt, and Procfile

 ## Model Change Log
 - converted the model to use dense layers instead of manualy doing matrix multiplication
 - added MRMSE loss function, refer [here](https://arxiv.org/pdf/1708.01715.pdf)


## Running on Server
to obtain prediction result from the server, post a request to http://hostip:5000/predict.
The request consists of a json with a 'user_ratings' element consisting an array of ratings given by the user to each restaurant. An example of this can be seen in client_example.py.
The server will return a json consisting a list of placeID sorted by its probability of being favored by the user and a list of probability for each restaurant.
