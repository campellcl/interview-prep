# interview-prep
Questions, answers, and resources for Machine Learning, Data Science, and Data Engineering interviews.

# Data Science Interview Guides
* https://towardsdatascience.com/mastering-the-data-science-interview-15f9c0a558a7
* https://www.analyticsvidhya.com/blog/2018/06/comprehensive-data-science-machine-learning-interview-guide/

# My Favorite Sites for Review of Topics:
* Brilliant: https://brilliant.org/

## Data Science Interview Questions Regarding Machine Learning
* NOTE: All questions are sourced from the following website, all answers are my own: https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/

1. You are given a train data set having 1000 columns and 1 million rows. The data set is based on a classification problem. Your manager has asked you to reduce the dimension of this data so that model computation time can be reduced. Your machine has memory constraints. What would you do? (You are free to make practical assumptions.)
    * First, as in most data science problems; I would conduct exploratory data analysis. I would generate scatter plots of the 1000 features and look for obvious correlations among related variables. If two variables are heavily correlated, then it may not be necessary to include them both. 
    * If it is too difficult to see any correlations due to the size of the dataset, sampling the dataset is always an option. However, care must be taken to draw a representative sample. 
    * Another approach would be a feature elimination method such as Recursive Feature Elimination (RFE). The downside to this approach is that RFE may take a while to run with 1000 features with limited RAM. If this is infeasible due to time constraints, then a dimensionality reduction technique such as Principle Component Analysis could be deemed more appropriate. That being said...
    * I would be hesitant to jump right into PCA as an initial approach without any exploratory analysis. Why? Because PCA is not magic, it operates under some restrictive assumptions (see: https://qr.ae/TWvWgC). 
        * PCA assumes that the principle components are linear combinations of the original features. If this is not the case, then PCA is a poor choice. 
        * PCA assumes that principle components are orthagonal. If the principle components are not orthogonal in the original feature space, PCA will perform poorly.  
        * PCA will yeild the principle components which capture the most variance individually. It may be the case that a combination of non-principle components in the original feature space do a better job of capturing the variance. PCA will neglect to take into account the impact of multiple non-principle components. 
2. Is rotation necessary in PCA? If yes, Why? What will happen if you don’t rotate the components?
    * Rotation is necessary in PCA because it is assumed that the principle components are orthagonal. If the components are not rotated, PCA will perform poorly. It is the orthagonality that maximizes the difference in variance between the two component axes. 
    * Recall that rotation does not change the relative distribution of the data, it just changes the physical location in space. Therefore it makes sense for PCA to rotate the data to maximize the difference in component axes before determining which components are principle. 
        * Note that if this rotation is not performed, then more principle components will need to be selected to capture the largest variances. 
3. You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?
   * One standard deviation captures ~68% of the data in a normal distribution (https://en.wikipedia.org/wiki/Standard_deviation#/media/File:Standard_deviation_diagram.svg). So ~32% of the data would not be effected. 
       * Here a normal distribution is assumed because the standard deviation is measured in relation to the median. 
4. You are given a data set on cancer detection. You’ve build a classification model and achieved an accuracy of 96%. Why shouldn’t you be happy with your model performance? What can you do about it?
    *  Accuracy is not the best metric for cancer detection. We should be more concerned with sensitivity and specificity metrics. It may be the case that it would be better to minimize the number of false positives (specificity). Diagnosing someone with cancer when they don't actually have cancer (a false positive) can be pretty devistating. 
    * NOTE: The source website (https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/) mentions truthfully that the cancer detection dataset is most likely an unbalanced classification problem. For unbalanced classification problems in general, accuracy is a poor choice in metric. A large percentage of the population does not have cancer, so we can assume this data set is inbalanced in regards to who does not have cancer. If this is the case, then obtaining a 96% accuracy is easy, if 96% of the population does not have cancer, saying no one has cancer will obtain the same result. What we really care about is detecting the minority class, the 4% of people who do have cancer. To this end, the measure of Sensitivity is more appropriate, as it takes into account the chances of having cancer to begin with (https://en.wikipedia.org/wiki/Sensitivity_and_specificity). 
    * What can we do about it? We can undersample or oversample the dataset to account for the inbalance. If we wish to tune the tradeoff between specificity and sensitivity, we can plot a Reciever Operating Characteristic (ROC) curve and vary the tradeoff between specificty and sensitivity. The Area Under the Curve (AUC) of the ROC will allow us to compare model performance, and select the best one for our particular use case in accordance to which is most important: specificity, sensitivity, or a preconcieved combination of both. 
5. Why is naive Bayes so ‘naive’?
    * Naive Bayes operates under the assumption that features in the data set are all equally important and statistically independent. This is rarely the case in real life, where often features are interrelated. 
6. Explain prior probability, likelihood and marginal likelihood in context of the Naive Bayes algorithm.
    * The prior probability (posterior) is the known (possibly pre-measured) probability of the event occuring. For instance, in the flip of a fair coin, the posterior probabilty of obtaining a heads is 0.5. This value can be identified emperically, or known in advance. But it is provided as-is without any additional pre-requisite knowledge. 
    * The likelihood is the probability (conditional probability) that an event (a measured set of successes) will occuring by knowing probability that an individual success will occur.
        * See: https://qr.ae/TWvWSY
    * In the context of naive bayes, the maximum likelihood is used as the output of the model. 
        * See: http://www.cs.columbia.edu/~mcollins/em.pdf
    * The marginal likelihood is the denominator of the standard Bayes theorem equation (https://qr.ae/TWvegC). For some practical interpretation of this, I like this answer (https://www.reddit.com/r/MachineLearning/comments/lirni/what_is_marginal_likelihood/). In short:
        * "Marginal likelihood is the expected probability of seeing the data over all the parameters theta, weighted appropriately by the prior. Bayes' law then says something like the conditional probability of a parameter at some value is the ratio of the likelihood of the data for that particular value over the expected likelihood from all values. Which is kind of intuitive as well."
7. You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. Can this happen? Why?
   * Sure. Decision trees are very dependent upon ordering. Asking the wrong question at the wrong time will cost you. Measures of purity and impurity are used to distinguish the 'quality' of a question/splitting criterion at each step in the decision tree. Without regard to these metrics, a decision tree can easily be useless; asking questions that fail to split the data. On the flip side, asking too many questions (adding too much depth to the tree) can cause everything to be discretized into singular bins; a clear sign of overfitting. 
      * NOTE: The authors of this question mention that time series data often possess linearity. Meanwhile decision trees are known to work best on non linear datasets. If the data set satisfies linearity assumptions, then a linear regression model may outperform a decision tree. 
8. You are assigned a new project which involves helping a food delivery company save more money. The problem is, company’s delivery team aren’t able to deliver food on time. As a result, their customers get unhappy. And, to keep them happy, they end up delivering food for free. Which machine learning algorithm can save them?
   * Oh man, this company has bigger problems (looking at you management). Who funded these guys? This is a classic route planning optimization scenario. AI algorithms might be a better fit then machine learning in this particular use case. There isn't really a compatible matrix of data, even if the routes are converted into a state transition table/matrix. I would probably turn to a reinforcement agent, or use one of the common solutions for the traveling salesman problem (which is really what this boils down too). It's been a while since AI. I don't have the algorithms memorized, but I sure know how to find them. A quick cursory look suggests nearest neighbors based optimization or ant colony optimization if looking to AI for solutions. I would need to revisit my AI notes, but I've done this before. 
9. You came to know that your model is suffering from low bias and high variance. Which algorithm should you use to tackle it? Why?
   * There isn't really a specific algorithm here, no free lunch theorem still applies. This is a classic bias vs. variance tradeoff problem. 
   * The variance in a model is the amount that the estimate of the target function will change if different training data was used (https://machinelearningmastery.com/gentle-introduction-to-the-bias-variance-trade-off-in-machine-learning/). 
   * The bias in a model is the underlying assumptions made about the form of the target function. For instance, Linear regression assumes a linear distribution of underlying data, this is highly biased (simplistic). 
      * To this end, a model with high variance indicates overfitting. A model with low bias indicates a relatively high degree of complexity. 
   * I would start by using a less complex model. Give the model less freedom to overfit (obtain high variance). Use cross validation to ensure that this is the case. If the model is complex (as in neural networks) give the network less neurons to learn with. Reduce the number of hyperplanes that the network can draw to seperate input into disctinct classes. 
   * NOTE: Another good approach is to apply regularization, penalize the higher model coefficients. The authors of these questions note that a bagging algorithm (such as random forest) may be appropriate here for modeling a high variance problem; especially when performed in an esemble with a voting metric. 
10. You are given a data set. The data set contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run PCA. Would you remove correlated variables first? Why?
   * Huh, good question. Not sure about this one. Intuitively, correlation is not inherently good for a dimensionality reduction technique that attempts to maximize variance (not covariance). But at the same time, the fact that several variables are highly correlated may aid the predictive power of the model. I think I would remove the correlated variables for the purpose of having a less complex model that can hopefully be just as capable with using only a subset of the correlated variables as is common in dimensionality reduction. 
      * NOTE: The authors of these questions point out (rightfully so) that adding correlated variables lets PCA put more importance on the correlated variables (which would exhibit twice the amount of variance as it would with uncorrelated variables). Hence the decision to remove the correlated variables prior to performing PCA is sound. 
11. After spending several hours, you are now anxious to build a high accuracy model. As a result, you build 5 GBM models, thinking a boosting algorithm would do the magic. Unfortunately, neither of models could perform better than benchmark score. Finally, you decided to combine those models. Though, ensembled models are known to return high accuracy, but you are unfortunate. Where did you miss?
12. How is kNN different from kmeans clustering?
13. How is True Positive Rate and Recall related? Write the equation.
14. You have built a multiple regression model. Your model R² isn’t as good as you wanted. For improvement, your remove the intercept term, your model R² becomes 0.8 from 0.3. Is it possible? How?
15. After analyzing the model, your manager has informed that your regression model is suffering from multicollinearity. How would you check if he’s true? Without losing any information, can you still build a better model?
16. When is Ridge regression favorable over Lasso regression?
17. Rise in global average temperature led to decrease in number of pirates around the world. Does that mean that decrease in number of pirates caused the climate change?
18. While working on a data set, how do you select important variables? Explain your methods.
19. What is the difference between covariance and correlation?
20. Is it possible capture the correlation between continuous and categorical variable? If yes, how?
21. Both being tree based algorithm, how is random forest different from Gradient boosting algorithm (GBM)?
22. Running a binary classification tree algorithm is the easy part. Do you know how does a tree splitting takes place i.e. how does the tree decide which variable to split at the root node and succeeding nodes?







































