# SMS Spam Classifier

This is an SMS Spam Classifier that predicts whether a text is spam or not and shows the confident score of that prediction.

For proper completion of this project, I divided this model into parts:

## Part 1: Importing and Understanding the Dataset

The dataset was called 'SMSSpamCollection' and it was gotten from a Kaggle Dataset. 

It was imported and peeked to understand the dataset details.

## Part 2: Cleaning the Dataset

The dataset was cleaned by lowering the case style and removing stopwords and noise.

## Part 3: Training and Testing the Model

The dataset was trained by using the train_split_model where, the messages were x and the labels y.

Using the x(input) -> y(label).

Where the model was trained on 80% of the data and tested on the unknown 20%.

For that training to occur, we started with a Bag of Words and the Multinominal Naive Bayes model.

The accuracy gotten was: 0.9820627802690582.

The classification report:

                precision    recall  f1-score   support

         ham       0.98      1.00      0.99       966
        spam       0.98      0.89      0.93       149

The precision was high but the recall was not high enough as the model could detect all ham messages but it couldn't not detect all spam messages as 149 * 0.89 = 133.

After checking the convolution matrix, 

[[963   3]
 [ 17 132]]

So we discover that we have 17 false positives and 3 false negatives.

The values in 17 false positives were brought out.

It was discovered that they had similar pattern with the legitimate messages the ham 