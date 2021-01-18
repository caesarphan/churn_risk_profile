# churn_risk_profile
customer (chrun) classification by months



Further improvements: better visualizations
Relativly low accuracy (upper ~80%)

labels:
0 --> High risk of churning
1 --> Medium Churn Risk
2 --> Low churn risk

Methodology:
10-fold CV
Hyper-Parameter tuning of following algorithms:
Random Forest
Naive Bayes
AdaBoost

Conclusion:
Naïve bayes appears to be the best model for identifying high-risk customers
In terms of recall, it far exceeds other algorithms with a score of 75% accuracy. While we are confident in the high precision, recall can be further improved. As seen in the confusion matrix, roughly 25% of customers who will churn are not accurately classified as at risk
Recency of purchase is the greatest indicator for churn risk
