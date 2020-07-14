library(h2o)
h2o.init()
file <- h2o.importFile("http://coursera.h2o.ai/cacao.882.csv")
x <- setdiff(colnames(file),c("Rating","REF","Review Date")) #Excluding Review Date and REF because it presumed to provide low information.
y <-  "Rating"
parts <- h2o.splitFrame(file,c(0.8,0.1))
train <- parts[[1]]
validate <- parts[[2]]
test <- parts[[3]]

system.time(baseLineModel <- h2o.deeplearning(x,y,train,validation_frame = validate,model_id = "BaseLine", seed = 123))
baseLineModel #Took 8.00 seconds 

plot(baseLineModel)

h2o.performance(baseLineModel,test)
#Important Stats about the baseLineModel when applied to the Test data set
# MSE:  0.1981134
# RMSE:  0.4450993
# MAE:  0.3637268
# RMSLE:  0.1099845
# Mean Residual Deviance :  0.1981134

#Now we will work on tuning our BaseLined Model (I used AutoML to do so. I ran it for an hour and used the parameters for deep learning given by autoML)
#As seen from the scoring history graph of the baseLineModel which is Deviance vs Epochs (Deviance because its a regression)
#There is not point in the graph where the curve flattens down or rises which is a good sign meaning there is no overfitting but could also imply that the deviance could be further decreased. 
#Hence we increased the epochs to see if we could get the lowest deviance value possible for the training and validation data set.

h2o.hist(train["Rating"]) #If you look at this graph, you will realise the distribution of the Rating data is not uniform and spikes in one point. 
#Hence we would use the laplace distribution to see if we could get a better result
#Also we would set variable_importance = TRUE, epsilon = 0.0001 to avoid division by zero during calculations, nfolds = 5 for more regularization, reduced hidden layers to 50 to avoid overfitting, fold assignment = Modulo to avoid even distribution of data sets, dropout ration of input and hidden = 0.1 and 0.5 respectively to avoid over-fitting

system.time(tunedModel <- h2o.deeplearning(x,y,train,validation_frame = validate,variable_importances = T,epochs = 478.4,hidden = c(50),distribution = "laplace",model_id = "Tuned",seed = 123,epsilon = 0.000001,fold_assignment = "Modulo",input_dropout_ratio = 0.1,hidden_dropout_ratios = 0.5,stopping_rounds = 0,stopping_tolerance = 0.02626128657194451,activation = "RectifierWithDropout",nfolds = 5))
tunedModel #The model took 78.10 seconds to train
plot(tunedModel) #The scoring history graph 

h2o.performance(tunedModel,test)
# The results when Tuned Model is applied to the test data set
# MSE:  0.1556883
# RMSE:  0.3945736
# MAE:  0.3200585
# RMSLE:  0.09861781
# Mean Residual Deviance :  0.1556883

h2o.saveModel(baseLineModel,path = "/mdl/") #Saving the baseLineModel on my local disk 
h2o.saveModel(tunedModel,"/mdl/") #Saving the tunedModel on my local disk



