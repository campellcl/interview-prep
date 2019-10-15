# Machine Learning Interview Preparation
## Python Machine Learning (Sabastian Rashka)
### Chapter Two: Training Machine Learning Algorithms for Classification (Perceptron):
1. What is the benefit of moving theta? What must the values of x_0 and w_0 be?
    * Moving theta allows us to compute the weights and the threshold 
    all at once using a single linear algebra expression.
    * X_0 = 1 and W_0 = -theta
2. An activation function takes the net input z= w^{T}x and squashes it into a binary output (-1 or 1). What is being
squashed? What is the range of z?
    * Z is a continuous variable which ranges from - infinity to + infinity.
    * Z is the "net-input" that is being squashed/quantized to be either -1
    or 1 in the case of the perceptron's activation function.
3. What is the learning rate and why is it important?
    * The learning rate controls the size of the updates to the weight
    vector with respect to the loss gradient.
    * The learning rate is important because it directly impacts the
    number of training epochs before convergence.
        * If the learning rate is too low, it may take forever to converge.
        * If the learning rate is too large, the algorithm may never converge, as it
        bounces around the bottom of the gradient parabola. 
4. Raschka writes, It is important to note that the convergence of the perceptron is only guaranteed if the two classes
 are linearly separable and the learning rate is sufficiently small." What does it mean to converge? Why can't it happen
 if there are misclassified samples?
    * In this case we are talking about the convergence of the gradient 
    descent algorithm toward what we hope is a global minima.
        * To be more specific, we are talking about the slope of the gradient function.
        At convergence, the slope should be near zero.
    * The perceptron learning algorithm is only guaranteed to converge if the data is linearly separable,
    and a small enough learning rate is used.
6. What is vectorization and why do we want it?
    * Vectorization is the process of converting code to use linear algebra operations on entire
    vectors instead of incrementally performing updates. 
    * Vectorization provides many benefits including: increased performance, 
    clearer to read code, and non-sequential variable updates.
7. What is the one-vs.-all technique and why do we use it?
    * The one-vs.-all technique is a method of extending a binary classifier for use with three or more classes.
    * A classifier is trained in each instance with knowledge of one class vs. all of the rest. This allows a simple
    binary classifier to perform multi-class classification.

### Chapter Two: Training Machine Learning Algorithms for Classification (Adaline):
1. According to the textbook what is the "main advantage" of the Adaline activation function compared to the setp function used in the Perceptron?
    * The Adaline activation function allows the algorithm to still learn from the examples that are predicted correctly,
    this is not the case in the Perceptron algorithm. 
2. What is the difference between the Perceptron and Adaline update rules?
3. The Perceptron `fit` method calls `predict` whereas the Adaline `fit` method calls `net_input`. Why?
    * In the case of calling `predict` we want the estimated target label. In the case of calling `net_input` we want a
    real value (continuous). The implications of this are somewhat severe. Adaline has an activation function and is
    trying to learn before quantizing the input. Perceptron quantizes the input and then tries to learn from it. 
        * The net impact of this is that the Perceptron algorithm can only ever get a distance of 0 or 2. It cannot
        learn from the samples it classifies correctly.
        * The Adaline algorithm on the other hand can learn from samples that it classifies correctly. If the positive
        sample is close to zero, Adaline will try and move the w vector so that positive samples get closer and closer 
        to positive 1. 
4. The Perceptron `fit` method has nested `for` loops whereas the Adaline `fit` method does not. How does this 
fundamentally change the learning algorithm?
    * The inner `for` loop in the Perceptron code updates w vector for every sample. Hence, it matters what order your
    samples are in the dataset. As the order dictates which samples actually change the w vector. Adaline does not have
    this constraint. 
    * During the Adaline algorithm we compute the dot-product between every row of X with it's corresponding error. By
    using the dot product, order does not matter. 

### Chapter Three: A tour of Machine Learning Classifiers Using Scikit-learn (Logistic Regression):
1. What is the No Free Lunch Theorem?
2. During feature scaling with the `sklearn.preprocessing`'s `StandardScalar` object, the `StandardScalar` is `fit` to
the training data. What does it need to compute in order to apply the transformation? Why is it important to apply the
transformation to the training and testing data?
3. Given the Logistic regression cost function. What is the cost for y=1 and phi(z)=1? What is the cost for y=1 and phi(z)
=1/e? What is the cost for y=1 and phi(z)=1/e^2?
4. Given the code for the AdalineGD classifier, what single function must you change in order for it to learn the coefficients
for the Logistic Regression classifier?
5. `lr = LogisticRegression(C=1000.0, random_state=0)` What is `C` and `random_state`? Where are `eta` and `n_iter`?
6. What is regularization? What is the problem with model coefficients which have a large magnitude?
7. What is L2 regularization? What does adding regularization do to the model's gradient?
 
### Chapter Three: A tour of Machine Learning Classifiers Using Scikit-learn (Linear SVM):
1. Why is Logistic Regression poorly named?
    * Logistic regression is a classification methodology, not a regression methodology. 
2. The optimization problem for SVM minimizes the length of the w vector under the following constraints MATH. Why do we
prefer a w vector with a small norm? What does it mean if the minimization is less than, equal to, or greater than 1, respectively?
3. We have discussed several linear classifiers: Perceptron, Adaline, Logistic Regression, and Linear Support Vector Machines. 
Given linearly separable classes, which classifiers are guaranteed to find a decision boundary that separates them?
4. We have discussed several linear classifiers: Perceptron, Adaline, Logistic Regression, and Linear Support Vector Machines. 
In general terms (no cost functions), how does each classifier define the "best" w vector?
5. We have discussed several linear classifiers: Perceptron, Adaline, Logistic Regression, and Linear Support Vector Machines. 
How does each classifier classify a new sample?
6. What is standardization and which classifiers benefit from it?
7. What is regularization and which classifiers benefit from it?
8. If you add a regularization term to the Adaline cost function, what do you think might happen?
9. Each of the classifiers (Perceptron, Adaline, Logistic Regression) use an activation function f(z) and compute errors
either before or after applying the quantizer. Write the equation for the activation function and "before" or "after" to
indicate when errors are computed for each of the classifiers specified above. 

### Chapter Three: A tour of Machine Learning Classifiers Using Scikit-learn (Nonlinear SVM, K-nearest Neighbors):
1. What is a kernel method? What role does it play?
2. What is the computational complexity of computing the dot product between two vectors?
3. What is the kernel trick? 
4. How many terms are there after mapping to a third-order polynomial from m=2 dimensions? What is the big-O complexity
of the dot-product in the higher dimensional space? What is the computational complexity after using the "kernel trick"?
5. Do we really have to compute the kernel function with every sample in the training set?
6. Show that the kernel function is equal to the dot-product of the mapping function for two-dimensional vectors. 
7. What is the default value of `k` in sklearn's implementation of K-Nearest Neighbors (KNN). Is this a good choice? Why
or why not?
8. How does KNN behave using k=1? What is the maximum value for k and how does it behave if you use it?

### Chapter Three: A tour of Machine Learning Classifiers Using Scikit-Learn (Decision Tree):
1. The goal of an SVM is to pick questions that separate the items into groups that are "pure". How is class "purity"
measured? How is the question decided which best splits the items?
2. What does it mean for a group of items to be pure in regards to a decision tree? Why is this a good thing?
3. If all of your test items ended up in the same leaf node of the decision tree, what would be the expected test accuracy?
Why?
4. In order to compute a split, we need the data stored in the parent node. Does this mean we need a copy of the data in
each node? Doesn't this require too much memory? Do we need to store the data after training?
5. How many features should be considered for a split? How many thresholds?
6. Why do you think Scikit-learn decision trees have exactly two children per parent?
7. Consider two extreme impurity measures. Impurity measure A is zero for perfectly pure nodes and one for everything else.
Impurity measure B is one for perfectly uniform class distributions, and zero everywhere else. Which impurity measure would
most prefer splits producing at least one perfectly pure child node? Based on this, what can you say about the preferences
for Entropy, Gini, and Misclassification rate? Why?
8. How complex is the decision tree shown in the image with selection boundaries? How could you make it more complex? How
could you make it less complex?
9. How does the Random Forest algorithm differ from a single decision tree?
10. Why would it be a good idea to combine multiple decision trees compared to training just one?

### Chapter Four: Building Good Training Sets -- Data Preprocessing:
1. What is a missing value? How might your data end up with them?
2. Why are missing values encoded as NaN? Are all NaNs missing values?
3. When dealing with strings as variables, when should you use each of these approaches: mapping to an integer, 
`OneHotEncoder`, `LabelEncoder`?
4. `StandardScaler` and `MinMaxScaler` are two of many ways to transform a feature. How can you decide which is most appropriate?
Give an example of a variable that would be well-suited to each of these transformations. 
5. What is the key difference between L1 and L2 regularization? When might you prefer one over the other?
6. Feature ranking, sequential backward selection, and recursive feature elimination are greedy approaches for selecting
a subset of features that perform well for classification. How could you search for the "best" feature set using an
approach that isn't greedy? What makes each of these methods more computationally efficient than the brute force approach?
What is the computational complexity of each approach?

### Chapter Five: Compressing Data via Dimensionality Reduction (Principal Component Analysis):
1. When does correlation equal covariance?
2. What are the seven steps of the PCA algorithm?
3. What is the purpose of Eigendecomposition? 
4. The eigenvector matrix is orthonormal, meaning that its columns are unit length and orthogonal (at a 90 degree angle)
to every other column. Show that V^{T}V is the identity matrix. 
5. Knowing that V is orthonormal and that VDV^{T} = cov(X), show that the covariance of XV is diagonal. What does it mean
for the covariance to be diagonal?
6. Give an example of a classifier that we've discussed that is affected by feature correlations and explain why. 

### Chapter Five: Compressing Data via Dimensionality Reduction (Linear Discriminant Analysis):
1. What are the two assumptions made by LDA?
2. For Within-Class Covariance, we don't want ot project onto the principal axes of the within-class covariance. Why not?
3. What is whitening?
4. LDA uses two covariance matrices, K_w and K_b. What information does each covariance matrix contain? In what direction
does the principal component (associated with the largest eigenvalue) of each scatter matrix point?

### Chapter Five: Compressing Data via Dimensionality Reduction (Kernel Principal Component Analysis):
* It is hard to ask questions about this one without mathematical typesetting and pictures. 

### Chapter Six: Learning Best Practices for Model Evaluation and Hyperparameter Tuning:
1. What does it mean for a model to have high bias? What about low bias?
2. What does it mean for a model to have high variance? What about low variance?
3. What is a hyperparameter, and how does it differ from a normal model parameter?
4. What is the process of selecting hte optimal hyperparameters for a given model called?
5. What is Dr. Parry's Unnamed Rule of Cross-Validation (multiple comparisons problem)?
6. The holdout method involves partitioning data into which three groups?
7. Why might it be a bad idea to use model performance computed on the testing data to optimize hyperparameters? In other
words, why bother creating a separate validation set?
8. Would it be a good idea to use k-fold cross-validation to find the "best" partition? That is, using hte fold that
produces the best performance as the new "validation" set and the others as the "training" set? Why or why not?
9. When k=10 for k-fold cross validation...How many models will be trained? How large will be the train set for each model?
How large will be the test set for each model?
10. How does the size of our dataset influence the choice of k in k-fold cross-validation?
11. How does k influence the bias of the performance estimate?
12. How does the computational complexity scale with k for k-fold cross-validation?
13. What special case of k-fold cross validation are we performing when k=n, where n is the number of training samples?
14. What is a key difference between k-fold cross-validation and stratified k-fold cross-validation?
15. What is the computational complexity of stratified k-fold cross-validation? How does this compare to normal k-fold
cross-validation?
16. When might we prefer stratified k-fold cross-validation to normal k-fold cross-validation?
17. What are possible ways to fix underfitting? What about overfitting?
18. What is a validation curve? How does it differ from a learning curve, and why is it useful?
19. Name some hyperparameters discussed this far. 
20. What determines the computational complexity of grid search?
21. How does nested cross-validation select the preferred model in the inner for-loop?
22. Say we used nested cross-validation to compare an SVM model to a Decision Tree Classifier and we received a nested
cross-validation performance of 97.9% for the SVM and 90.8% for the Decision Tree. What conclusion can we make
about these models if both were provided with the same sample data? Which algorithm would perform better given a
new subset of the sample data?
23. How does the computational complexity of nested cross-validation compare to normal k-fold cross validation?
24. Is there ever a case where we would prefer k-fold cross-validation over nested cross-validation?
25. Your cancer screening strategy achieves 99% accuracy on the general population where 99% of people don't have this
type of cancer. Is this a good performance? Why or why not?
26. You are tasked with creating a cancer screening method with 100% sensitivity. What does it mean when a screening method
achieves 100% sensitivity? Is it possible? If so, how?
27. You are tasked with creating a cancer screening method with 100% specificity. What does it mean when a screening method
achieves 100% specificity? Is it possible? If so, how?
28. What does it mean when a classifier has an ROC curve that is on the main diagonal line? What if it's below the diagonal line?
29. Where would a perfect classifier fall on an ROC curve?
30. How would we compare two different ROC curves to determine which model performed the best?
31. What information would a precision-recall curve provide that an ROC curve does not?
32. When scoring multiclass classifiers, when might we prefer micro-averaging to macro-averaging and vice versa?
