# interview-prep
Questions, answers, and resources for Machine Learning, Data Science, and Data Engineering interviews.

# Data Science Interview Guides
* https://towardsdatascience.com/mastering-the-data-science-interview-15f9c0a558a7
* https://www.analyticsvidhya.com/blog/2018/06/comprehensive-data-science-machine-learning-interview-guide/

# My Favorite Sites for Review of Topics:
* For Mathematics (Statistics, Probability, Linear Algebra):
   * Brilliant: https://brilliant.org/
   * Khan Academy: https://www.khanacademy.org/
* For Programming In General:
  * Hacker Rank: https://www.hackerrank.com/
  * Leet Code: https://leetcode.com/
* For SQL:
  * SQL ZOO: https://sqlzoo.net/
* For Machine Learning:
  * 
# My Favorite Textbooks and Resources for Learning These Topics Initially
* For Statistics:
  * Elements of Statistical Learning: https://web.stanford.edu/~hastie/ElemStatLearn/
  * Explained Visually: http://setosa.io/ev/
* For Data Mining:
  * Introduction to Data Mining: https://www-users.cs.umn.edu/~kumar001/dmbook/index.php
* For Machine Learning: 
  * NOTE: There are honestly too many honerable mentions to list them all... but I'll give it a shot. 
  * Python Machine Learning: https://sebastianraschka.com/books.html
  * The Deep Learning Book: https://www.deeplearningbook.org/
* For Artificial Intelligence:
  * Artificial Intelligence: A Modern Approach: http://aima.cs.berkeley.edu/

# The Best Instructors for This Material (Hands Down):
* NOTE: Simply google: <topic you are stuck on> <instructor name>
* Andrew Ng
* Geoffery Hinton
* Sebastian Thrun & Peter Norvig

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
   * GBM stands for Gradient Boosting Machine. I am not familiar with gradient boosting, but have read about it briefly here: https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab. Looks like these kinds of models would be prone to overfitting. An ensemble of overfitted models, is still going to overfit. Perhaps the features are not representative of the target function, or the size of the dataset is too small to learn a meaningful signal?
   * The authors point out that since all five GBM models have no accuracy improvement, it suggests that the models are correlated. Hence, all models provide roughly the same information. An ensemble of correlated models is a poor choice. 
12. How is kNN different from kmeans clustering?
   * K-Nearest Neighbors uses only the nearest neighbors. K-Means uses a nearest centroid to cluster. Hence K-Means is a better choice if data is naturally distributed in clumps around a series of centroids. If data appears to be randomly distributed, perhaps KNN is a better choice, as KNN allows for complex decision boundaries to be drawn with relatively low choices in K. 
      * NOTE: I missed the mark on this one. The authors point out that Kmeans is unsupervised, whereas KNN is a classification (or regression) algorithm. This is a topic I should review. Most of the data I work with is considered supervised learning.  
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
23. You’ve built a random forest model with 10000 trees. You got delighted after getting training error as 0.00. But, the validation error is 34.23. What is going on? Haven’t you trained your model perfectly?
24. You’ve got a data set to work having p (no. of variable) > n (no. of observation). Why is OLS as bad option to work with? Which techniques would be best to use? Why?
25. What is convex hull ? (Hint: Think SVM)
26. We know that one hot encoding increasing the dimensionality of a data set. But, label encoding doesn’t. How ?
27. What cross validation technique would you use on time series data set? Is it k-fold or LOOCV?
28. You are given a data set consisting of variables having more than 30% missing values? Let’s say, out of 50 variables, 8 variables have missing values higher than 30%. How will you deal with them?
29. ‘People who bought this, also bought…’ recommendations seen on amazon is a result of which algorithm?
30. What do you understand by Type I vs Type II error?
31. You are working on a classification problem. For validation purposes, you’ve randomly sampled the training data set into train and validation. You are confident that your model will work incredibly well on unseen data since your validation accuracy is high. However, you get shocked after getting poor test accuracy. What went wrong?
32. You have been asked to evaluate a regression model based on R², adjusted R² and tolerance. What will be your criteria?
33. In k-means or kNN, we use euclidean distance to calculate the distance between nearest neighbors. Why not manhattan distance?
34. Explain machine learning to me like a 5 year old.
35. I know that a linear regression model is generally evaluated using Adjusted R² or F value. How would you evaluate a logistic regression model?
36. Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?
37. Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?
38. When does regularization becomes necessary in Machine Learning?
39. What do you understand by Bias Variance trade off?
40. OLS is to linear regression. Maximum likelihood is to logistic regression. Explain the statement.

## Data Science Questions Regarding Probability (https://www.analyticsvidhya.com/blog/2017/04/40-questions-on-probability-for-all-aspiring-data-scientists/)
1) Let A and B be events on the same sample space, with P (A) = 0.6 and P (B) = 0.7. Can these two events be disjoint?
   A) Yes
   B) No
2) Alice has 2 kids and one of them is a girl. What is the probability that the other child is also a girl? You can assume that there are an equal number of males and females in the world.
A) 0.5
B) 0.25
C) 0.333
D) 0.75
3) A fair six-sided die is rolled twice. What is the probability of getting 2 on the first roll and not getting 4 on the second roll?
A) 1/36
B) 1/18
C) 5/36
D) 1/6
E) 1/3
4) P(A U B U C) = P(A n C^{c}) + P(C) + P(B n A^{c} n C^{c})
True or False?
A) True
B) False
5) Consider a tetrahedral die and roll it twice. What is the probability that the number on the first roll is strictly higher than the number on the second roll? Note: A tetrahedral die has only four sides (1, 2, 3 and 4).
A) 1/2
B) 3/8
C) 7/16
D) 9/16
6) Which of the following options cannot be the probability of any event? 
A) -0.00001
B) 0.5
C) 1.001
7) Anita randomly picks 4 cards from a deck of 52-cards and places them back into the deck ( Any set of 4 cards is equally likely ). Then, Babita randomly chooses 8 cards out of the same deck ( Any set of 8 cards is equally likely). Assume that the choice of 4 cards by Anita and the choice of 8 cards by Babita are independent. What is the probability that all 4 cards chosen by Anita are in the set of 8 cards chosen by Babita?
A){}^{48}C_{4} x {}^{52}C_{4}
B){}^{48}C_{4} x {}^{52}C_{8}
C){}^{48}C_{8} x {}^{52}C_{8}
D) None of the above
8) A player is randomly dealt a sequence of 13 cards from a deck of 52-cards. All sequences of 13 cards are equally likely. In an equivalent model, the cards are chosen and dealt one at a time. When choosing a card, the dealer is equally likely to pick any of the cards that remain in the deck. If you dealt 13 cards, what is the probability that the 13th card is a King?
A) 1/52
B) 1/13
C) 1/26
D) 1/12
9) A fair six-sided die is rolled 6 times. What is the probability of getting all outcomes as unique?
A) 0.01543
B) 0.01993
C) 0.23148
D) 0.03333
10) A group of 60 students is randomly split into 3 classes of equal size. All partitions are equally likely. Jack and Jill are two students belonging to that group. What is the probability that Jack and Jill will end up in the same class?
A) 1/3
B) 19/59
C) 18/58
D) 1/2
11) We have two coins, A and B. For each toss of coin A, the probability of getting head is 1/2 and for each toss of coin B, the probability of getting Heads is 1/3. All tosses of the same coin are independent. We select a coin at random and toss it till we get a head. The probability of selecting coin A is ¼ and coin B is 3/4. What is the expected number of tosses to get the first heads?
A) 2.75
B) 3.35
C) 4.13
D) 5.33
12) Suppose a life insurance company sells a $240,000 one year term life insurance policy to a 25-year old female for $210. The probability that the female survives the year is .999592. Find the expected value of this policy for the insurance company.
A) $131
B) $140
C) $112
D) $125
13) P(A n B n C^{c}) = P(A)P(C^{c} n A | A) P(B | A n C^{c})
True or False?
A) True
B) False
14) When an event A independent of itself?
A) Always
B) If and only if P(A)=0
C) If and only if P(A)=1
D) If and only if P(A)=0 or 1
15) Suppose you’re in the final round of “Let’s make a deal” game show and you are supposed to choose from three doors – 1, 2 & 3. One of the three doors has a car behind it and other two doors have goats. Let’s say you choose Door 1 and the host opens Door 3 which has a goat behind it. To assure the probability of your win, which of the following options would you choose.
A) Switch your choice
B) Retain your choice
C) It doesn’t matter probability of winning or losing is the same with or without revealing one door
16) Cross-fertilizing a red and a white flower produces red flowers 25% of the time. Now we cross-fertilize five pairs of red and white flowers and produce five offspring. What is the probability that there are no red flower plants in the five offspring? 
A) 23.7%
B) 37.2%
C) 22.5%
D) 27.3%
17) A roulette wheel has 38 slots – 18 red, 18 black, and 2 green. You play five games and always bet on red slots. How many games can you expect to win?
A) 1.1165
B) 2.3684
C) 2.6316
D) 4.7368
18) A roulette wheel has 38 slots, 18 are red, 18 are black, and 2 are green. You play five games and always bet on red. What is the probability that you win all the 5 games?
A) 0.0368
B) 0.0238
C) 0.0526
D) 0.0473
19) Some test scores follow a normal distribution with a mean of 18 and a standard deviation of 6. What proportion of test takers have scored between 18 and 24?
A) 20%
B) 22%
C) 34%
D) None of the above
20) A jar contains 4 marbles. 3 Red & 1 white. Two marbles are drawn with replacement after each draw. What is the probability that the same color marble is drawn twice?
A) 1/2
B) 1/3
C) 5/8
D) 1/8
21) Which of the following events is most likely? 
A) At least one 6, when 6 dice are rolled
B) At least 2 sixes when 12 dice are rolled
C) At least 3 sixes when 18 dice are rolled
D) All the above have same probability
22) Suppose you were interviewed for a technical role. 50% of the people who sat for the first interview received the call for second interview. 95% of the people who got a call for second interview felt good about their first interview. 75% of people who did not receive a second call, also felt good about their first interview. If you felt good after your first interview, what is the probability that you will receive a second interview call?
A) 66%
B) 56%
C) 75%
D) 85%
23) A coin of diameter 1-inches is thrown on a table covered with a grid of lines each two inches apart. What is the probability that the coin lands inside a square without touching any of the lines of the grid? You can assume that the person throwing has no skill in throwing the coin and is throwing it randomly. 
A) 1/2
B) 1/4
C) Π/3
D) 1/3
24) There are a total of 8 bows of 2 each of green, yellow, orange & red. In how many ways can you select 1 bow? 
A) 1
B) 2
C) 4
D) 8
25) Consider the following probability density function: What is the probability for X≤6 i.e. P(x≤6)
A) 0.3935
B) 0.5276
C) 0.1341
D) 0.4724
26) In a class of 30 students, approximately what is the probability that two of the students have their birthday on the same day (defined by same day and month) (assuming it’s not a leap year)?
For example – Students with birthday 3rd Jan 1993 and 3rd Jan 1994 would be a favorable event.
A) 49%
B) 52%
C) 70%
D) 35%
27) Ahmed is playing a lottery game where he must pick 2 numbers from 0 to 9 followed by an English alphabet (from 26-letters). He may choose the same number both times.
If his ticket matches the 2 numbers and 1 letter drawn in order, he wins the grand prize and receives $10405. If just his letter matches but one or both of the numbers do not match, he wins $100. Under any other circumstance, he wins nothing. The game costs him $5 to play. Suppose he has chosen 04R to play.
What is the expected net profit from playing this ticket?
A) $-2.81
B) $2.81
C) $-1.82
D) $1.82
28) Assume you sell sandwiches. 70% people choose egg, and the rest choose chicken. What is the probability of selling 2 egg sandwiches to the next 3 customers?
A) 0.343
B) 0.063
C) 0.44
D) 0.027
29) HIV is still a very scary disease to even get tested for. The US military tests its recruits for HIV when they are recruited. They are tested on three rounds of Elisa( an HIV test) before they are termed to be positive.
The prior probability of anyone having HIV is 0.00148. The true positive rate for Elisa is 93% and the true negative rate is 99%.
What is the probability that a recruit has HIV, given he tested positive on first Elisa test? The prior probability of anyone having HIV is 0.00148. The true positive rate for Elisa is 93% and the true negative rate is 99%.
A) 12%
B) 80%
C) 42%
D) 14%
30) What is the probability of having HIV, given he tested positive on Elisa the second time as well.
The prior probability of anyone having HIV is 0.00148. The true positive rate for Elisa is 93% and the true negative rate is 99%.
A) 20%
B) 42%
C) 93%
D) 88%
31) Suppose you’re playing a game in which we toss a fair coin multiple times. You have already lost thrice where you guessed heads but a tails appeared. Which of the below statements would be correct in this case?
A) You should guess heads again since the tails has already occurred thrice and its more likely for heads to occur now
B) You should say tails because guessing heads is not making you win
C) You have the same probability of winning in guessing either, hence whatever you guess there is just a 50-50 chance of winning or losing
D) None of these
32) The inference using the frequentist approach will always yield the same result as the Bayesian approach.
A) TRUE
B) FALSE
33) Hospital records show that 75% of patients suffering from a disease die due to that disease. What is the probability that 4 out of the 6 randomly selected patients recover?
A) 0.17798
B) 0.13184
C) 0.03295
D) 0.35596
34) The students of a particular class were given two tests for evaluation. Twenty-five percent of the class cleared both the tests and forty-five percent of the students were able to clear the first test.
Calculate the percentage of students who passed the second test given that they were also able to pass the first test.
A) 25%
B) 42%
C) 55%
D) 45%
35) While it is said that the probabilities of having a boy or a girl are the same, let’s assume that the actual probability of having a boy is slightly higher at 0.51. Suppose a couple plans to have 3 children. What is the probability that exactly 2 of them will be boys?
A) 0.38
B) 0.48
C) 0.58
D) 0.68
E) 0.78
36) Heights of 10 year-olds, regardless of gender, closely follow a normal distribution with mean 55 inches and standard deviation 6 inches. Which of the following is true?
A) We would expect more number of 10 year-olds to be shorter than 55 inches than the number of them who are taller than 55 inches
B) Roughly 95% of 10 year-olds are between 37 and 73 inches tall
C) A 10-year-old who is 65 inches tall would be considered more unusual than a 10-year-old who is 45 inches tall
D) None of these
37) About 30% of human twins are identical, and the rest are fraternal. Identical twins are necessarily the same sex, half are males and the other half are females. One-quarter of fraternal twins are both males, one-quarter both female, and one-half are mixed: one male, one female. You have just become a parent of twins and are told they are both girls. Given this information, what is the probability that they are identical?
A) 50%
B) 72%
C) 46%
D) 33%
38) Rob has fever and the doctor suspects it to be typhoid. To be sure, the doctor wants to conduct the test. The test results positive when the patient actually has typhoid 80% of the time. The test gives positive when the patient does not have typhoid 10% of the time. If 1% of the population has typhoid, what is the probability that Rob has typhoid provided he tested positive?
A) 12%
B) 7%
C) 25%
D) 31.5%
39) Jack is having two coins in his hand. Out of the two coins, one is a real coin and the second one is a faulty one with Tails on both sides. He blindfolds himself to choose a random coin and tosses it in the air. The coin falls down with Tails facing upwards. What is the probability that this tail is shown by the faulty coin?
A) 1/3
B) 2/3
C) 1/2
D) 1/4
40) A fly has a life between 4-6 days. What is the probability that the fly will die at exactly 5 days?
A) 1/2
B) 1/4
C) 1/3
D) 0

























































