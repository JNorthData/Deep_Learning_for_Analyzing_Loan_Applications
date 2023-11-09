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


The first run of my model produced an accuracy score of only .73, so further enhancement and evaluation is needed. 

# optimization 1
Since CLASSIFICATION produced the most sparse columns when one-hot encoded, I filtered out the bottom 3% of value-counts, which reduced inputs from 83 to 46, almost half. Still, this resulted in poor accuracy, with Loss = .55 and Accuracy = .73

In fact, after this first adjustment there were still noticeably more sparce columns (95% of values were zeros) than non-sparse columns. This suggests that Primary Component Analysis might be helpful. I also decided to apply Early Stopping since it seemed that many epochs would be required, which could be costly in terms of time. This required that I redo some earlier code, because I had to split our original data into 3 sets: training, validation, and testing. 

# optimization 2
PCA determined that only 34 PCs were needed to account for 95% of variation in my data. However, testing the model using 34 PCs yeilded no improvement. I spent a little more time here playing with parameters such as batch, epoch, and learning rate. I determined that a 1024 batch with 10000 epochs at a learning rate of .001 was a good baseline for basic tuning, especially with early stopping. With patience set to 1000, Most runs stopped out at around 1000 epochs, achieving only Loss=.53 and Accuracy=.74. Essentially, I found that I could run the model 50 times or 1000 times, and it simply didn't affect the results. The structure of the model itself must need some reworking. 

# optimization 3
Since I wanted to reduce dimensionality using PCA - specifically to avoid making any biased decisions as to which features to drop - there wasn't much about my preprocessing that I wanted to change. In fact, I even created a second notebook and removed all but the essential encoding and scaling steps, and still the Loss was near .55 and the Accuracy near .73. In the process, I found some errors in my code, and now had a cleaner model, but still the Loss/Accuracy numbers would not budge!

I decided to try adding another hidden layer, with half the neurons of the first; still no change. I added a 3rd, and then a 4th and 5th layer, and finally 10 hidden layers (!), increasing neurons until I ended up with this:

        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        dense (Dense)               (None, 34)                1190      
                                                                        
        dense_1 (Dense)             (None, 68)                2380      
                                                                        
        dense_2 (Dense)             (None, 102)               7038      
                                                                        
        dense_3 (Dense)             (None, 102)               10506     
                                                                        
        dense_4 (Dense)             (None, 102)               10506     
                                                                        
        dense_5 (Dense)             (None, 102)               10506     
                                                                        
        dense_6 (Dense)             (None, 68)                7004      
                                                                        
        dense_7 (Dense)             (None, 68)                4692      
                                                                        
        dense_8 (Dense)             (None, 68)                4692      
                                                                        
        dense_9 (Dense)             (None, 34)                2346      
                                                                        
        dense_10 (Dense)            (None, 17)                595       
        ...
        Total params: 61473 (240.13 KB)
        Trainable params: 61473 (240.13 KB)
        Non-trainable params: 0 (0.00 Byte)


...still, .55 Loss and .73 Accuracy. . 

# optimization 4
At this point I decided not to spend any more time on this project, as I now had a good understanding of what it takes and how to go about optimizing a neural network model. Since PCA did not seem to yeild any benefit, it was dropped from the model. I also reduced the number of hidden layers since increasing layers didn't help either. I used some other activation functions also. 

If I were to resume optimization, I would next attempt to reduce dimensionality (because of the encoding) by determining feature importance and selecting unimportant features to remove from the dataset.













