library(h2o)
h2o.init()
data <- h2o.importFile("http://coursera.h2o.ai/house_data.3487.csv")
year <- as.integer(sapply(data$date,function(ele)substr(ele,1,4))) #Extracts year from Date Column
month <-  as.integer(sapply(data$date , function(ele)substr(ele,5,6))) #Extracts month from Date Column
day <- as.integer(sapply(data$date , function(ele)substr(ele,7,8))) #Extracts day from Date Column
d <- as.data.frame(data) #Converted h2o object into R data frame for further manipulation
deal <- cbind(d,year,month,day) #Deal is a data frame combing all columns 
colnames(deal)
relevant <- setdiff(colnames(deal),c("id","date")) #Removing Id and Date because they would add unnecessary noise to the data set 
working = deal[,relevant] #Has all relevant columns 
data1 <- as.h2o(working) #Loading working to h2o cluster 
h2o.describe(data1) #Checking for cardinality or missing data, but no missing data neither cardinality was there 
x <- setdiff(colnames(data1),"price") #Extracting all independent dimensions 
y <- "price" #Price is to be predicted as a dependent dimension
parts <- h2o.splitFrame(data1,0.9,seed = 123)
train <- parts[[1]]
h2o.hist(train$price) #Saw the distribution of price and most of the prices seems to be around a certain number. Hence Guassion distribution would work for the models
test <- parts[[2]]
nrow(train)
nrow(test)

#For inspiration, I used AutoML 
dl <- h2o.deeplearning(x,y,train,nfolds = 5,fold_assignment = "modulo",hidden = c(200,200,200),stopping_tolerance = 0.007168136484453069,keep_cross_validation_predictions = T,epochs = 40)
dl #In the deep learning model I added an additional layer of hidden neuron or nodes so that more weights could be added to the model. Even I increased the epochs to 40 because the scoring history plot for 10 epochs was declining but showed signs of improvement. 
gbm <- h2o.gbm(x,y,train,nfolds = 5,fold_assignment = "Modulo",ntrees = 140 , max_depth = 6, min_rows = 1,keep_cross_validation_predictions = TRUE)
gbm #I increased the max_depth to 6 and number of trees to 140 to allow more training but reduce overfitting by setting min_rows = 1, wiz. the minimum number of weights to be added per node. The default of min_row is 10.
rdm <- h2o.randomForest(x,y,train,nfolds = 5, fold_assignment = "Modulo",ntrees = 140 , max_depth = 6, min_rows = 1,keep_cross_validation_predictions = TRUE)
rdm #Followed similar steps as mentioned in the gradient boosting machine model.
glm <- h2o.glm(x,y,train,nfolds = 5, fold_assignment = "Modulo",keep_cross_validation_predictions = TRUE)
glm #Kept default settings 

#Now lets make a stacked Ensemble
Final_Model <- h2o.stackedEnsemble(x,y,train,base_models = c(gbm@model_id,rdm@model_id,glm@model_id,dl@model_id))
Final_Prediction <- h2o.performance(Final_Model,test) #RMSE = 113561.5 wiz. lesser than 123000 , this is the best model
Final_Prediction

#It's time to save the models
h2o.saveModel(dl, path = "/FA/")
h2o.saveModel(gbm,path = "/FA/")
h2o.saveModel(rdm,path = "/FA/")
h2o.saveModel(glm,path = "/FA/")
h2o.saveModel(Final_Model,path = "/FA/")
