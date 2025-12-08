# Concept Check
To avoid data leakage, I made sure the model never saw information from the future. 
Since this is a time-based dataset, I kept the rows in chronological order and used the 
most recent 20% of the timeline as the test set. This mirrors a real prediction scenario, 
where the goal is to forecast future months instead of randomly shuffled data.

The lag features I created (such as the previous month’s trash weight) only use 
information that would have actually been available at prediction time, so they do not 
leak any future values into the model.

For evaluation, I focused on RMSE and MAE because they show how far the predictions are 
from the true values in tons. I also used R² to measure how much of the variation in the 
target the model can explain.

I first trained a Linear Regression model as a baseline. It performed extremely well 
on the test set, partly because the lag features created a very strong linear pattern 
in the data. However, this almost perfec performance can be misleading, since 
Linear Regression assumes the relationships will always stay linear and stable. If 
the pattern shifts or becomes more irregular, a simple linear model may fail.

Because of this, I trained a Random Forest model as well. Even though its results 
were slightly less “perfect” numerically, it handled non-linear patterns much better 
and was more robust to unusual months or sudden spikes. Random Forest also works 
naturally with SHAP, which allowed me to interpret the model more clearly. For these 
reasons better flexibility, and interpretability I selected the 
Random Forest as the final model, using 200 trees with a fixed random_state for 
reproducibility.
