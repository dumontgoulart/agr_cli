**Quick reproductible setup for hyperparameter tunning using gridsearch CV for random forest model.
**

Steps:
  1 - Among the many variables , select the most important (not shown);
  2 - Divide the data into two parts: Train and Test;
  3 - Use the train data under a stratified cross validation process (5 splits, 6 random reshuffles) to search the grid for best paramters;
  4 - Define the best paratmeters based on the accuracy score;
  5 - Evaluate performance at the Test data using following scores: 1) Accuracy, 2) Precision, 3) Recall, 4) F-1 Score.
