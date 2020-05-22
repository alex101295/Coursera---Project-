# ENVIRONMENT PREPARATION - Downloading necessary libraries for the programmation

# Set the working directory
setwd("/Users/alexandralocchi/Documents/EDHEC NICE/Cours/FE")

library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(corrplot)
library(rattle)
library(randomForest)
library(RColorBrewer)
set.seed(1012)

# DATA LOADING & CLEANSING 

# Set URL for the download 
UrlTrain <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
UrlTest  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Download adequate datasets
training <- read.csv(url(UrlTrain))
testing  <- read.csv(url(UrlTest))

# Create a partition with the training dataset 
inTrain  <- createDataPartition(training$classe, p=0.7, list=FALSE)
TrainSet <- training[inTrain, ]
TestSet  <- training[-inTrain, ]
dim(TrainSet)
## [1] 13737   160
dim(TestSet)
## [1] 5885  160

#The two datasets (TrainSet & TestSet) have multiple NA numbers and near zero variance variables. I will remove both.
nzv_var <- nearZeroVar(TrainSet)
TrainSet <- TrainSet[, -nzv_var]
TestSet <- TestSet[, -nzv_var]
dim(TrainSet)
## [1] 13737    104
dim(TestSet)
## [1] 5885    104
na_var <- sapply(TrainSet, function(x) mean(is.na(x))) > 0.95
TrainSet <- TrainSet[ , na_var == FALSE]
TestSet <- TestSet [ , na_var == FALSE]
dim(TrainSet)
## [1] 13737    59
dim(TestSet)
## [1] 5885   59

# I will also remove columns 1 to 5 because they are identification variations 
TrainSet <- TrainSet[ , -(1:5)]
TestSet  <- TestSet [ , -(1:5)]
dim(TrainSet)
## [1] 13737    54
dim(TestSet)
## [1] 5885   54


# CORRELATION ANALYSIS 
corr_matrix <- cor(TrainSet[ , -54])
corrplot(corr_matrix, order = "FPC", method = "circle", type = "lower",
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))


# PREDICTION MODEL BUILDING 

  # DECISION TREE MODEL
set.seed(1012)
fit_dec_tree <- rpart(classe ~ ., data = TrainSet, method="class")
fancyRpartPlot(fit_dec_tree)
# Prediction of the decision tree model on TestSet
predict_dec_tree <- predict(fit_dec_tree, newdata = TestSet, type="class")
conf_matrix_dec_tree <- confusionMatrix(predict_dec_tree, TestSet$classe)
conf_matrix_dec_tree
# Ploting the matrix results of the decision tree model
plot(conf_matrix_dec_tree$table, col = conf_matrix_dec_tree$byClass, 
     main = paste("Decision Tree Accuracy: Predictive Accuracy =",
                  round(conf_matrix_dec_tree$overall['Accuracy'], 4)))

# GENERALIZED BOOSTED MODEL
set.seed(1012)
ctrl_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
fit_GBM  <- train(classe ~ ., data = TrainSet, method = "gbm",
                  trControl = ctrl_GBM, verbose = FALSE)
fit_GBM$finalModel
# Prediction of the decision tree model on TestSet
predict_GBM <- predict(fit_GBM, newdata=TestSet)
conf_matrix_GBM <- confusionMatrix(predict_GBM, TestSet$classe)
conf_matrix_GBM
# Ploting the matrix results of the GBM model
plot(conf_matrix_GBM$table, col = conf_matrix_GBM$byClass, 
     main = paste("GBM - Accuracy =", round(conf_matrix_GBM$overall['Accuracy'], 4)))

  # RANDOM FOREST MODEL
set.seed(1012)
ctrl_RF <- trainControl(method="cv", number=3, verboseIter=FALSE)
fit_RF <- train(classe ~ ., data=TrainSet, method="rf",
                          trControl=ctrl_RF)
fit_RF$finalModel
# Prediction of the decision tree model on TestSet
predict_RF <- predict(fit_RF, newdata = TestSet)
conf_matrix_RF <- confusionMatrix(predict_RF, TestSet$classe)
conf_matrix_RF
# Ploting the matrix results of the random forest model
plot(conf_matrix_RF$table, col = conf_matrix_RF$byClass, 
     main = paste("Random Forest: Predictive Accuracy =",
                  round(conf_matrix_dec_tree$overall['Accuracy'], 4)))


# CCL - APPLY THE SELECTED MODEL TO THE DATA TEST
predict_Test <- predict(fit_RF, newdata=testing)
predict_Test

