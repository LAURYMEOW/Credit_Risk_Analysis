# Credit Risk Analysis

#### Objetive:

The goal is to perform a credit card risk analysis to assess the risk of fraud.
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we need to use different techniques to train and evaluate models with unbalanced classes.
For this process, the unbalanced learning and scikit-learn libraries will be used to build and evaluate models through resampling. The credit data set of credit cards that LendingClub has provided to us will be used.

#### Results:

For this analysis we have been made six competitive models to evaluate which of them is the one that best represents the data and at the same time choose which one can help us to make a good analysis of them.
The measures that we will be reviewing in each model will be: Accuracy score, Precision, Recall and F1.

The first two are based on Oversampling. The idea behind these models is thatif one class has too few instances in the training set, we choose more instances from that class for training until it'slarger.

#### - The Random Over Sample model: In this model instances of the minority class are radomly selected and added to the training set until the majority and minority classes are balanced.
The results to this model are given at the table below:

#### Table 1: 

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/Random_Oversample_table.png)

In this first model there is evidence that we have two completely unbalanced classes. Due to the above, it is expected that the highest class will have a greater weight, which indicates low risk. This can be seen in the accuracy test result.
However, the recall test, which is the fraction of positives that were correctly identified, tells us that about half of both classes were correctly identified.
The other indicator that we are interested in evaluating is the F1 that responds to: What percent of positive predictions were correct?. Our model says that it predicted 74% of low risk correctly but only 0.02% of high risk.
This last indicator suggests that the model is not useful for our purpose.


#### - The SMOTE model: Here like the after model, the size of the minority is increased. In this model, new instances are interpolated, i.e. new values are created until the majority and minority classes are balanced.
The results to this model are given at the table 2:

#### Table 2:

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/SMOTE%20Oversample.png)

In this model we can't see any improve on the metrics of the high risk class over the Random Oversampling model results a pesar de que el accuracy score tuvo una ligera mejoria.
 
The following model belongs to the category of Under sample model. In these model the size of the majority class is decreased. #### Here we only uses actual data.

#### -The Cluster Centroids model: This model identifies clusters of the majority class the generates synthetic data points, called centroids, that are representative of the clusters. Then majority class is then undersampled down to the size of the minority class.
In the table 3 we can see the model results.

#### Table 3:

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/Cluster%20centroids%20table.png)

These results are worse than the two oversampling models before. Even the accuracy score is low.

The fourth model is a combinatorial approach of over- and undersampling using the SMOTEENN algorithm.

#### - The SMOTEEN model: This model combines the SMOTE and Edited Nearest Neighbors algorithms and consist in two-step process. First oversample the minority class with SMOTE then clean the result data with undersampling. This last step remove some class's outliers from the dataset which allows us to work with a cleaner database. 
The results to this model are in the Table 4.

#### Table 4:

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/Combination%20sampling.png)

This model shows a slight improvement in the metrics compared to the three previous models, specifically in the Accuracy Score and the Recall test. Despite the above, this model cannot be considered a good model for our objective.


The last two models are based on ensemble learning, which fundamental idea is that two is better than one. That is the process of combining multiple models that reduce bias. 


#### -Balanced Random Forest Classifier: It is a ensemble method in which each tree of the forest will be provided a balanced bootstrap sample. I.e. A balanced random forest randomly under-samples each boostrap sample to balance it.
Let's go to look the results.

#### Table 5:

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/Balanced%20Random%20Forest%20Classifier.png)

We can see better results in this model. However, we can still observe that the precision is still very low, which indicates that there are a large number of false positives. 
The recall test is not as low as the accuracy test but it is not very reliable because the accuracy score and the F1 test remain low.


#### -Easy Ensemble Classifier: The classifier is an ensemble of AdaBoost learners trained on different balanced boostrap samples. The balancing is achieved by random under-sampling.

* In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This process is repeated until the error is minimized. 
The results for our last model are in the table below.

#### Table 6:

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Module-17-Challenge-Resources/Resources/Easy%20Ensemble%20AdaBoost.png)

Definitely out of the six models this is the best performer with a high accuracy score of 93% and an F1 test of 19%. This last result is likely because the low value to the precision test.
If we consider that the model is more sensitive than accurate, we can interpret the results of the Recall test as follows: the model was able to identify 91% of positive cases for high risk, although we also have a large number of false positives.
In other words, the model manages to identify all high-risk cases but also identifies other low-risk cases as high-risk.

Although this last model can be useful, it can still be improved to have a more accurate and reliable model.

#### Summary 

We have evaluated our data with 3 balancing techniques and with 2 ensemble classifiers. Coming to the conclusion that an ensemble model greatly improves the results for a very unbalanced model.

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Plotting%20test%20and%20models.png)

And also for the case at hand, measuring credit risk, the model that behaved best is the Easy Ensemble Classifier. Although it is the most competitive model, there are still improvements to be made to be able to use it.

![](https://github.com/LAURYMEOW/Credit_Risk_Analysis/blob/main/Plot%20Easy%20ensemble%20model%20test.png)

One of them is to review the most significant variables for the model and evaluate if it is theoretically appropriate to discard the least significant ones.
This is possible considering that a model with more variables than it needs is overfitting. It may also be that there are variables that generate multicolnearity. Due to the above, it is important to make a more detailed analysis of the important variables for the model.





