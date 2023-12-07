library(ranger)
library(caret)
library(data.table)
cc_data <- read.csv("C:\\Users\\satti\\OneDrive\\Desktop\\creditcard.csv")
dim(cc_data)
head(cc_data,10)
tail(cc_data,10)
table(cc_data$Class)
summary(cc_data$Amount)
names(cc_data)
var(cc_data$Amount)
sd(cc_data$Amount)
head(cc_data)
cc_data$Amount=scale(cc_data$Amount)
d1=cc_data[,-c(1)]
head(d1)


#DataModelling
library(caTools)
set.seed(123)
ds = sample.split(d1$Class,SplitRatio=0.80)
datafortrain = subset(d1,ds==TRUE)
datafortest = subset(d1,ds==FALSE)
dim(datafortrain)
dim(datafortest)


#LogisticRegressionModel
LogisticModel = glm(Class~.,datafortest,family=binomial())
summary(LogisticModel)
plot(LogisticModel)

Logistic_Model_train = glm(Class~.,datafortrain,family=binomial())
summary(Logistic_Model_train)

library(pROC)
lr.predict <- predict(Logistic_Model_train,datafortest, probability = TRUE)
auc.gbm = roc(datafortest$Class, lr.predict, plot = TRUE, col = "red")


#Decision Tree Model
library(rpart) 
library(rpart.plot)
decisionTreeModel <- rpart(Class ~ . , cc_data, method = 'class')
predictedvalue <- predict(decisionTreeModel, cc_data, type = 'class')
prob <- predict(decisionTreeModel, cc_data, type = 'prob')

rpart.plot(decisionTreeModel)

#ANN

library(neuralnet)
ANNModel <- neuralnet(Class~.,data=head(datafortrain,50000),linear.output=F)
plot(ANNModel)

predictedANN=compute(ANNModel,datafortest)
result_ANN=predictedANN$net.result
result_ANN=ifelse(result_ANN>0.5,1,0)


#Gradient Boosting

library(gbm, quietly=TRUE)

system.time(
  gbmModel <- gbm(Class ~ .
                  , distribution = "bernoulli"
                  , data = rbind(datafortrain, datafortest)
                  , n.trees = 500
                  , interaction.depth = 3
                  , n.minobsinnode = 100
                  , shrinkage = 0.01
                  , bag.fraction = 0.5
                  , train.fraction = nrow(datafortrain) / (nrow(datafortrain) + nrow(datafortest))
  )
)

gbm.iter = gbm.perf(gbmModel, method = "test")


model.influence = relative.influence(gbmModel, n.trees = gbm.iter, sort. = TRUE)

plot(gbmModel)


testgbm = predict(gbmModel, newdata = datafortest, n.trees = gbm.iter)
gbmauc = roc(datafortest$Class, testgbm, plot = TRUE, col = "green")
print(gbmauc)
