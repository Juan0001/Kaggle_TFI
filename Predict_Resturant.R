## import libraries
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(dplyr)
library(readr)
library(ggplot2)
library(reshape)

## read the data
train <- read.csv("train.csv")
test <- read.csv("test.csv")

## make histogram of the variables that will be used in the prediction
data <- melt(train[,-c(1:5)])
g <- ggplot(data,aes(x = value))
p <- g + facet_wrap(~variable,scales = "free_x") + geom_histogram()

## make a log value of the revenue
train$LogRev <- log(train$revenue)

## delete the first five colomns of the data that will not be used in the prediction
train <- select(train, -(1:5))
train <- select(train, -revenue)

## CART model
## Cross-validation

# Number of folds
tr.control = trainControl(method = "cv", number = 10)

# cp values
cp.grid = expand.grid( .cp = (0:10)*0.001)

cvTree = train(LogRev ~ ., data=train, method="rpart", trControl = tr.control, tuneGrid = cp.grid)

# Extract tree
best.tree = cvTree$finalModel

# Prediction
best.tree.pred = predict(best.tree, newdata=test)

# change back the logged revenue
test.pred <- exp(best.tree.pred)

submissionCART <- data.frame(Id=test$Id, Prediction=test.pred)
write.csv(submissionCART, "submissionCART.csv", row.names=FALSE)

## Random forest model
library(randomForest)
revenueRF <- randomForest(LogRev ~ ., data=train)
predictRF <- predict(revenueRF, newdata=test)

test.RF <- exp(predictRF)

submissionRF <- data.frame(Id=test$Id, Prediction=test.RF)
write.csv(submissionRF, "submissionRF.csv", row.names=FALSE)

## RF, ntree=500
revenueRF1 <- randomForest(LogRev ~ ., data=train, ntree=500)
predictRF1 <- predict(revenueRF1, newdata=test)
test.RF1 <- exp(predictRF1)
submissionRF1 <- data.frame(Id=test$Id, Prediction=test.RF1)
write.csv(submissionRF1, "submissionRF1.csv", row.names=FALSE)

## RF, ntree=4000
revenueRF2 <- randomForest(LogRev ~ ., data=train, ntree=4000)
predictRF2 <- predict(revenueRF2, newdata=test)
test.RF2 <- exp(predictRF2)
submissionRF2 <- data.frame(Id=test$Id, Prediction=test.RF2)
write.csv(submissionRF2, "submissionRF2.csv", row.names=FALSE)