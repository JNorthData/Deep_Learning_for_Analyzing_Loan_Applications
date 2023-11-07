# deep-learning-challenge

The challenge is to create a binary classification neural network model that will predict the probability  of success for businesses that are applying for loans, based on previous patterns in loan application data from our Alphabet Soup venture capital organization. 

Our dataset consists of over 34,000 samples of applications for loans that were granted to fund businesses, and is labeled by the success/failure outcome of those loans. (80% of the samples were used for training, with 20% reserved for testing our model)

The original data used to train our model had 10 features:
    APPLICATION_TYPE
    AFFILIATION   
    CLASSIFICATION      
    USE_CASE               
    ORGANIZATION     
    STATUS         
    INCOME_AMT      
    SPECIAL_CONSIDERATIONS   
    ASK_AMT 
    and 
    IS_SUCCESSFUL used to label our training and test data.    

After encoding all classification features, the resulting scaled dataset consisted of 84 columns, which were all used as inputs. 

A Sequential neural network model was chosen for our basic feed-forward classification task. The first instance of this model was structured as follows:

- An input layer with 83 inputs, using a ReLu activation for 83 neurons. 
- 1 hidden layer, also using ReLu activation for 42 (half of inputs) neurons. 
- An output layer using Sigmoid activation for only 1 ouput neuron, providing a success/failure prediction.

- Optimized using Adam, for ease of handling many parameters. 
- Loss function is 'binary_crossentropy', suitable for our binary classification. 


