# Machine Learning Engineer Nanodegree
## Capstone Proposal
Stephen O'Kennedy  
25 April, 2018

## Proposal: Identify and classify toxic online comments

### Domain Background

Since it’s inception the internet has allowed people from most parts of the world to freely communicate, debate, and collaborate with each other over a wide range of topics and projects. Platforms like Github, Hackernews, Twitter, Wikipedia, etc. form the foundations for which these interactions can take place. Many of these communities have standards and rules in place to facilitate conversations, and to prevent these communities from being hijacked, or destroyed by toxic behaviour. It is becoming increasingly harder to regulate and enforce these standards. In fact  Facebook are currently hiring more and more moderators to sift through questionable content [1].

[Conversation AI](https://conversationai.github.io/) are working to provide tools to help improve online conversation[2][2]. One area that they’re focusing on is the study of negative online behaviours, like toxic comments[2][2]. As their Kaggle page states, the current models in use for detecting toxic comments still make errors, and they don’t allow users to be able to identify the types of toxicity they’re interested in finding. For example some platforms may be fine with comments that contain profanities.

### Problem Statement

Given a dataset that contains a large number of Wikipedia comments which have been labelled by human raters for toxic behaviour.  We want to create a model that predicts the probability of different types of toxicity for each comment.

### Datasets and Inputs

We are provided with a dataset in csv format where we have the following columns:
`id`
`comment_text` : [String]
`toxic ` : int
`severe_toxic ` : int
`obscene ` : int
`threat` : int
`insult ` : int
`identity_hate` : int

The `comment_text` column is comment that we want to feed into our classifier, and the outputs  will be will be a vector containing the probabilities of the comment being one of the `toxic`, `obscene`, etc.	  The training data set can be found here [3].


### Solution Statement

The solution will involved the development of deep learning algorithm that uses Keras with TensorFlow being used as the backend. Our aim is to use a multi-class CNN to process the content of the comments and out put a ROC AUC score [4][4]. Finally, predictions will be made on the test data set and will be evaluated on Kaggle.

### Benchmark Model
The benchmark score we’ll use to compare our model against will  be `0.982900`. This score was is in the 50th percentile of the public leaderboards[5][5], and was calculated using ROC AUC metric [4][4].

### Evaluation Metrics

Submissions are evaluated by using the ROC AUC metric. Each comment in the test data set will need to be labeled with the predictions for each type of toxicity appearing in each comment, and will need to be submitted to Kaggle.

ROC is the receiver operating characteristic curve. It is a graphical plot that displays the discrimination threshold of a binary classier, which is what we’ll need to build. Our threshold ($T$), which is used to classify a datapoint as either positive or negative, is by default set $0.5$. We take the true positive rate ($TPR$) and false positive rate ($FPR$) for all scores and plot a curve. Calculating the AUC (area under the curve) will reduce the curve down to a single value, $1 \ge A < 0$. Where $A$ is the AUC. If $A$ is close to $1.0$ we’ve got a perfect classifier, if However it is $0.5$ of lower than our classifier is doing little more than guessing [6].

The formula is:

$$
A = \int_{\infty}^{-\infty} TPR(T)FPR'(T)dT = \int_{-\infty}^{\infty}\int_{\infty}^{-\infty}I(T' > T)f_1(T')f_0(T)dT'dT = P(X_1 >X_0)
$$


### Project Design

As described in the problem statement, we will be analysing comments made on wikipedia. We will first need to perform data analysis and get familiar with the data. There are several questions that come to mind immediately that I want to understand about the data. For example, are the classes balanced? Is the data set complete? Do some of the comments use non standard characters?

Once we've completed our data analysis we will begin text processing. this will include removing stop words, stemming, tokenising, vectorising, etc. We would like to point out since we're trying to process natural language we're going to try different n-gram ranges such as (2,4) in order to capture toxic phrases. It also worth noting that we'll need to investigate how to deal with misspellings. Therefore just by processing the text content of the contents alone would lead us to implement use grid search to tune the our text vectorising algorithm.

We will then need to define our CNN or MLP architecture. Our initial plan is to create a reasonably shallow neural network and establish a baseline model to work from. We model our initial network on something like Alexnet. We could then use a pre-trained network such as Google's inception or VGG19. This raises question then, is our training data big enough for us to retrain our neural network and fine-tune? Or do we remove the initial convoluted layers, add our own, and bolt on the rest of the network and slice the end of the network to match the number of outputs.


[1]: http://fortune.com/2018/03/22/human-moderators-facebook-youtube-twitter/
[2]: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
[3]: https://www.kaggle.com/c/8076/download/train.csv.zip
[4]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
[5]: https://www.kaggle.com/c/8076/publicleaderboarddata.zip
[6]: https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
