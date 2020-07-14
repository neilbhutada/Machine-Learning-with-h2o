# I am going to make a data frame called "HeavyWeightBoxer".
# I am going to make a machine learning model using Random Forest/GBM to identify whether a person is a heavy weight boxer or not.

#First let me create the data frame (data set)
weight <- runif(500, min = 40 , max = 120) #Weight is considered to be in kgs
height <- runif(500 , min = 150 ,  max = 210) #Height is considered to be in cms

boxer <- c("yes" , "no") #We will convert this into a factor later on with 500 entries 
for(i in 1:500)
{
  if(weight[i]>=110)   #If their weight is more than 110 kgs then they are a heavy weight boxer
  {
    boxer[i] = "yes"
  }
  else
  {
    boxer[i] = "no"
  }
}
boxer <- as.factor(boxer) #converting into a factor
# You can see the current values of the columns by executing the lines below: 
boxer
weight
height 
#Now we will combine all these three columns to make a data frame called HeavyWeightBoxer 
HeavyWeightBoxer <- data.frame(weight_in_kgs = weight , height_in_cms = height , heavy_weight_boxer = boxer )
#See the value of HeavyWeightBoxer
HeavyWeightBoxer

library(h2o) #Loading the h2o library in our environment 
h2o.init()

dataset <- as.h2o(HeavyWeightBoxer)  #Loading our data in h2o 
partition <- h2o.splitFrame(dataset , 0.8) #Splitting data 
train <- partition[[1]]
test <- partition[[2]]

goodModel <- h2o.randomForest( 1:2 ,3, train ,nfolds = 5 , model_id = "Assignment/GoodModel/Used RandomForest")
goodModel #We can understand the model is good model with no overfitting by checking the scoring history graph, AUC of ROC (wiz. equal to 1) , comparing RMSE of the validation and training data side by side in h2o flow
# For bad model we will add some noise to the data and use gbm to allow overfitting
#We will add some noise to the data by randomly converting some of the no values to yes in boxer vector
noise_adder = c("yes" , "no")
for(i in 1:500)
{
  if(boxer[i] == "yes")
  {
    boxer[i] = sample(noise_adder , 1)
  }
}
HeavyWeightBoxer <- data.frame(weight_in_kgs = weight , height_in_cms = height , heavy_weight_boxer = boxer )
#Now we will perform some intiall calls again
dataset <- as.h2o(HeavyWeightBoxer)  #Loading our data in h2o 
partition <- h2o.splitFrame(dataset , 0.8) #Splitting data 
train <- partition[[1]]
test <- partition[[2]]

badModel <- h2o.gbm(1:2 , 3, train, nfolds = 5 , ntrees = 100 , max_depth = 25 , model_id = "Assignment/BadModel/UsedGBM")
badModel
#We overfitted the model by 1) Adding noise to the data , 2) Use GBM instead of Random Forest , 3) Increasing the number of trees and maximum depth
#You could see the ROC , scoring history and RMSE of the training and validation data side by side in h2o flow to verify overfitting 
