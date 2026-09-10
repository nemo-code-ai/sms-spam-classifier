# SMS Spam Classifier

## Introduction

This project is an SMS Spam Classifier that uses machine learning to predict whether an SMS message is **spam** or **legitimate (ham)**.

The application also provides the model's **spam probability**, showing how strongly the model believes a message is spam.

Rather than training one model and stopping there, I treated this project as an experiment. I tested different feature representations, machine learning models, class weights, n-grams, and prediction thresholds to understand what actually worked best for this dataset.

The final model uses:

* **Bag of Words (BoW)**
* **Balanced Logistic Regression**
* **0.5 prediction threshold**

---

# Part 1: Importing and Understanding the Dataset

The dataset used for this project is the **SMS Spam Collection**, which contains SMS messages labelled as either `ham` or `spam`.

The dataset was loaded and inspected to understand its structure and class distribution.

The important columns were:

* `message` → the SMS text
* `label` → whether the message is `ham` or `spam`

The dataset is imbalanced because there are significantly more legitimate messages than spam messages.

This became important later when evaluating the model.

---

# Part 2: Cleaning the Dataset

Before training the models, the messages were cleaned to make them easier for the machine learning algorithms to process.

The main preprocessing step was converting the messages to lowercase and removing unnecessary punctuation/noise.

For example:

```text
"CONGRATULATIONS! You Won A Prize!"
```

becomes something closer to:

```text
"congratulations you won a prize"
```

The goal was to make similar words appear as the same feature to the model.

---

# Part 3: Training and Testing the Models

The dataset was split into training and testing sets.

The messages were treated as the input `X`, while the spam/ham labels were treated as the target `y`.

```text
X (message) → Model → y (spam/ham)
```

The dataset was split into:

* **80% training data**
* **20% testing data**

The model was trained using the training data and then evaluated on the unseen test data.

## Experiment 1: Bag of Words + Multinomial Naive Bayes

I started with a **Bag of Words (BoW)** representation.

Bag of Words converts text into numerical features based on the words that appear in the dataset.

For example:

```text
"free prize"
```

could become a numerical representation based on whether words such as `free`, `prize`, and others appear.

The first model used was **Multinomial Naive Bayes**.

### Results

**Accuracy: 98.21%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.98 |   1.00 |     0.99 |
| Spam  |      0.98 |   0.89 |     0.93 |

The model performed very well overall.

However, spam recall was **89%**, meaning the model detected about 89% of the spam messages.

The confusion matrix was:

```text
[[963   3]
 [ 17 132]]
```

The matrix follows this structure:

```text
[[True Negative, False Positive],
 [False Negative, True Positive]]
```

Therefore:

* True Negatives = 963
* False Positives = 3
* False Negatives = 17
* True Positives = 132

The 17 false negatives were particularly important because they represented spam messages that the model incorrectly classified as legitimate.

I inspected these incorrect predictions to understand why the model was making these mistakes.

Some of the spam messages had patterns that looked similar to legitimate conversations, making them difficult for a simple word-based representation to distinguish.

---

## Experiment 2: TF-IDF + Multinomial Naive Bayes

Next, I replaced Bag of Words with **TF-IDF**.

TF-IDF gives different importance to words based on how frequently they appear across documents. In theory, this can help the model focus more on informative words.

However, the results were worse.

### Results

**Accuracy: 95.43%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.95 |   1.00 |     0.97 |
| Spam  |      1.00 |   0.66 |     0.79 |

Confusion matrix:

```text
[[966   0]
 [ 51  98]]
```

Spam recall dropped from **89% to 66%**.

This was an important lesson:

> A more sophisticated feature representation does not automatically produce a better model.

For this particular dataset and model combination, **Bag of Words performed better than TF-IDF**.

---

## Experiment 3: Bag of Words + Logistic Regression

I then changed the machine learning algorithm from Multinomial Naive Bayes to **Logistic Regression** while keeping Bag of Words.

### Results

**Accuracy: 98.30%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.98 |   1.00 |     0.99 |
| Spam  |      1.00 |   0.87 |     0.93 |

Confusion matrix:

```text
[[966   0]
 [ 19 130]]
```

Compared with the original Naive Bayes model, spam recall decreased slightly from **89% to 87%**.

So although Logistic Regression performed extremely well, it did not solve the false-negative problem.

---

## Experiment 4: TF-IDF + Logistic Regression

I also tested Logistic Regression with TF-IDF.

### Results

**Accuracy: 97.04%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.97 |   1.00 |     0.98 |
| Spam  |      1.00 |   0.78 |     0.88 |

Confusion matrix:

```text
[[966   0]
 [ 33 116]]
```

Again, TF-IDF performed worse than Bag of Words for this project.

At this point, the experiments suggested that **Bag of Words was the better feature representation for this dataset**.

---

# Experiment 5: Bag of Words + Balanced Logistic Regression

The dataset contains many more ham messages than spam messages.

Because of this imbalance, I wanted to make the Logistic Regression model pay more attention to the minority class.

I used:

```python
class_weight = "balanced"
```

This automatically adjusts the importance of the classes during training.

### Results

**Accuracy: 98.74%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.99 |   1.00 |     0.99 |
| Spam  |      0.99 |   0.91 |     0.95 |

Confusion matrix:

```text
[[965   1]
 [ 13 136]]
```

This was a significant improvement.

Spam recall increased from **87% to 91%**, while maintaining **99% precision**.

However, the model still missed 13 spam messages.

That led to the next question:

**Why were these messages still being classified incorrectly?**

---

# Experiment 6: N-Grams

One possible problem with Bag of Words is that it mainly considers individual words.

For example:

```text
"claim your prize"
```

is represented through individual words such as:

```text
claim
your
prize
```

But sometimes the combination of words can carry more meaning.

So I tested **n-grams**, allowing the model to consider combinations of words.

For example:

* Unigram → `claim`
* Bigram → `claim prize`
* Trigram → `claim your prize`

I tested Bag of Words with unigrams and bigrams using Balanced Logistic Regression.

### Results

**Accuracy: 98.39%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.98 |   1.00 |     0.99 |
| Spam  |      0.99 |   0.89 |     0.94 |

Confusion matrix:

```text
[[965   1]
 [ 17 132]]
```

Spam recall dropped from **91% to 89%**.

Therefore, adding n-grams did not improve the final model.

---

# Experiment 7: TF-IDF + N-Grams

I also tested another variation using TF-IDF with the n-gram representation.

### Results

**Accuracy: 97.40%**

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| Ham   |      0.99 |   0.98 |     0.98 |
| Spam  |      0.89 |   0.92 |     0.90 |

Confusion matrix:

```text
[[949  17]
 [ 12 137]]
```

Spam recall increased to **92%**, but spam precision dropped to **89%**.

This meant the model was catching more spam, but it was also incorrectly flagging more legitimate messages as spam.

For this application, I considered that tradeoff too costly.

Therefore, I kept **Bag of Words + Balanced Logistic Regression** as the best model.

---

# Part 4: Threshold Tuning

The Logistic Regression model does not simply produce a label. It can also produce a probability representing how strongly it believes a message belongs to the spam class.

By default, the model uses a threshold of **0.5**:

```text
Spam probability >= 0.5 → SPAM
Spam probability < 0.5 → HAM
```

However, changing the threshold changes the balance between precision and recall.

Because the final model still had **13 false negatives out of 149 spam messages**, the false-negative rate was:

```text
13 / 149 ≈ 8.72%
```

I tested thresholds from **0.2 to 0.8** to see whether the model could catch more spam without creating too many false positives.

| Threshold | Spam Precision | Spam Recall |  Spam F1 |
| --------: | -------------: | ----------: | -------: |
|       0.2 |           0.82 |    **0.95** |     0.88 |
|       0.3 |           0.93 |        0.92 |     0.93 |
|       0.4 |           0.97 |        0.92 |     0.94 |
|   **0.5** |       **0.99** |        0.91 | **0.95** |
|       0.6 |           0.99 |        0.90 |     0.94 |
|       0.7 |           1.00 |        0.89 |     0.94 |
|       0.8 |           1.00 |        0.85 |     0.92 |

Lowering the threshold generally allowed the model to classify more messages as spam, increasing recall but also increasing false positives.

At first, I considered using **0.4** because it increased spam recall from 91% to 92% while maintaining 97% precision.

However, after comparing all the metrics, I chose **0.5**.

The 0.5 threshold produced the highest spam F1-score in my experiments:

```text
Precision: 0.99
Recall:    0.91
F1-score:  0.95
```

Therefore, the final threshold remained **0.5**.

---

# Part 5: Implementing the Model

After selecting the final model, I saved both the trained Logistic Regression model and the fitted Bag of Words vectorizer using `joblib`.

The final pipeline is:

```text
SMS Message
     ↓
Text Cleaning
     ↓
Bag of Words
     ↓
Balanced Logistic Regression
     ↓
Spam Probability
     ↓
0.5 Threshold
     ↓
HAM / SPAM
```

The model was then integrated into a **Streamlit application** so that users can enter an SMS message and receive a prediction.

The application displays whether the message is:

```text
SPAM
```

or

```text
HAM
```

along with the model's spam probability.

---

# Final Model

After all of the experiments, the final model was:

**Bag of Words + Balanced Logistic Regression + 0.5 Threshold**

Its performance on the test set was:

| Metric    | Spam |
| --------- | ---: |
| Precision | 0.99 |
| Recall    | 0.91 |
| F1-score  | 0.95 |

Overall test accuracy:

**98.74%**

The biggest lesson from this project was that improving a machine learning system is not simply about choosing the most advanced technique.

I tested several approaches, investigated where the model failed, changed the representation and model, experimented with class imbalance, and finally tuned the decision threshold.

The experiments showed me that **the best solution depends on the data and the problem, not on which technique sounds more advanced.**

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Model

**Final Model:** Balanced Logistic Regression
**Feature Extraction:** Bag of Words
**Threshold:** 0.5
