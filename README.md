# interview-prep
Questions, answers, and resources for Machine Learning, Data Science, and Data Engineering interviews.

# Data Science Interview Guides
* https://www.analyticsvidhya.com/blog/2018/06/comprehensive-data-science-machine-learning-interview-guide/

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
3. You are given a data set. The data set has missing values which spread along 1 standard deviation from the median. What percentage of data would remain unaffected? Why?
4. You are given a data set on cancer detection. You’ve build a classification model and achieved an accuracy of 96%. Why shouldn’t you be happy with your model performance? What can you do about it?
5. Why is naive Bayes so ‘naive’?
6. Explain prior probability, likelihood and marginal likelihood in context of naiveBayes algorithm?
7. You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. Can this happen? Why?
8. You are assigned a new project which involves helping a food delivery company save more money. The problem is, company’s delivery team aren’t able to deliver food on time. As a result, their customers get unhappy. And, to keep them happy, they end up delivering food for free. Which machine learning algorithm can save them?
9. You came to know that your model is suffering from low bias and high variance. Which algorithm should you use to tackle it? Why?
10. You are given a data set. The data set contains many variables, some of which are highly correlated and you know about it. Your manager has asked you to run PCA. Would you remove correlated variables first? Why?
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







































