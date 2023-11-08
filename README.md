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

- Optimized using Adam, for ease of handling many parameters. This was later replaced with Keras Legacy, in order to better accomodate my M1 Mac machine. 
- Loss function is 'binary_crossentropy', suitable for our binary classification. 
- A metric of Accuracy is used for feedback during the training and evaluation phase. 


The first run of our model produced an accuracy score of only .73, so further enhancement and evaluation is needed. 

Since CLASSIFICATION produced the most sparse columns when one-hot encoded, I filtered out the bottom 3% of value-counts, which reduced inputs from 83 to 46, almost half. Still, this resulted in poor accuracy, with Loss = .55 and Accuracy = .73

In fact, after this first adjustment there were still noticeably more sparce columns (95% of values were zeros) than non-sparse columns. This suggests that Primary Component Analysis might be helpful. I also decided to apply Early Stopping since it seemed that many epochs would be required, which could be costly in terms of time. This required that I redo some earlier code, because I had to split our original data into 3 sets: training, validation, and testing. 

PCA determined that only 34 PCs were needed to account for 95% of variation in our dataset. However, testing the model using 34 PCs yeilded no improvement. I spent a little more time here playing with parameters such as batch, epoch, and learning rate. 




