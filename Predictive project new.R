bc=read.csv("C:/Users/mteig/OneDrive/Desktop/MA 5790/data (1).csv")
str(bc)
#Removing the last column with no value
bc=bc[,-33]
str(bc)
numeric_variables <- bc[, sapply(bc, is.numeric)]
dim(numeric_variables)
########################################################################################
#Check for missing values
missing_values <- colSums(is.na(numeric_variables)) 
missing_values
#Plot needed for missing values
# Calculate missing values for each column
barplot(missing_values, main = "Missing Values by Variable", xlab = "Variables",
        ylab = "Number of Missing Values", col = "skyblue", names.arg = names(missing_values))


# Check distribution - histograms
# Create histograms for the numeric variables
par(mfrow=c(2, 3))  # Set the layout to display multiple plots on one page
for (col in colnames(numeric_variables)) {
  hist(numeric_variables[[col]], main = col, col = "skyblue", xlab = col, ylab = "Frequency")
}



#Check skewnes values
library(e1071)
# Select only the numeric variables (excluding 'id' and 'X')
numeric_variables <- bc[, sapply(bc, is.numeric)]
# Remove the 'id' and 'X' columns from the numeric variables
numeric_variables <- numeric_variables[, !(names(numeric_variables) %in% c("id", "X"))]
# Calculate skewness for each numeric variable
skewness_values <- sapply(numeric_variables, skewness)
# Print the skewness values
print(skewness_values)

dim(numeric_variables)

########################################################################################
# Check for outliers
# Set up a grid for multiple boxplots
# Create boxplots for the numeric variables
par(mfrow=c(2, 3))  # Set the layout to display multiple plots on one page
for (col in colnames(numeric_variables)) {
  boxplot(numeric_variables[[col]], main = col, col = "skyblue")
}
######################################################################################
#Correlation between variables
correlations=cor(numeric_variables)
dim(correlations)
correlations[1:4, 1:4]
library(corrplot)
corrplot(correlations, order = "hclust")

#####################################################################################
#Data transformation (boxcox and spatialsign)
library(caret)
trans <- preProcess(numeric_variables, method = c("BoxCox", "center", "scale", "spatialSign"))
trans
# Apply the transformations
transformed <- predict(trans, numeric_variables)
head(transformed)
dim(transformed)
dim(numeric_variables)

#Plot boxplot after transformation
par(mfrow=c(3, 3))  
lapply(names(transformed), function(x) {boxplot(transformed[, x], main=x, ylab=x)})

# After transformation Create histograms for the numeric variables
par(mfrow=c(2, 3))  # Set the layout to display multiple plots on one page
for (col in colnames(transformed)) {
  hist(transformed[[col]], main = col, col = "skyblue", xlab = col, ylab = "Frequency")
}

#Print out skewed values
library(e1071)
skewness_values_trans <- apply(transformed,2, skewness)
print(skewness_values_trans)

#PCA transformation reduce multicolinearity
pcatrans <- preProcess(transformed, method=c('pca'))
#predict
transformed_predictors=predict(pcatrans,transformed)
dim(transformed_predictors)

# Scree plot to visualize variance explained by each principal component
pca_result <- prcomp(transformed, center = TRUE, scale. = TRUE)
plot(pca_result, type = "l", main = "Scree Plot")

#Correlation between variables after PCA transformation
correlations_trans=cor(transformed_predictors)
dim(correlations_trans)
library(corrplot)
corrplot(correlations_trans, order = "hclust")
table(bc$diagnosis)
nearZeroVar(bc$diagnosis)
#Plot of the diagnosis column
# Create a barplot for the "diagnosis" column
barplot(table(bc$diagnosis), main = "Diagnosis Barplot", xlab = "Diagnosis", 
        ylab = "Count", col = c("skyblue", "grey"))

# Add labels to the bars
text(
  barplot(table(bc$diagnosis), plot = FALSE),
  table(bc$diagnosis),
  pos = 3,
  cex = 0.8
)

dim(transformed_predictors)

set.seed(100)
response <- bc$diagnosis
head(response)
#combined_data <- cbind(response,transformed_predictors)


trainIndex <- createDataPartition(y = response, p = .80, list = FALSE)
trainresponse <- response[trainIndex]
testresponse <- response[-trainIndex]

trainPredictors <- transformed_predictors[trainIndex, ]
testPredictors <- transformed_predictors[-trainIndex, ]

####### Nonlinear Discriminant Analysis ##########

ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)

set.seed(476)
ndaFit <- train(trainPredictors,
                y = trainresponse,
                method = "mda",
                metric = "ROC",
                tuneGrid = expand.grid(.subclasses = 1:20),
                trControl = ctrl)
ndaFit
plot(ndaFit)

# Predictions and performance evaluation
predictions <- predict(ndaFit, testPredictors)
predict(ndaFit, newdata = testPredictors, type = "prob")

postResample(predictions,testResponse)
confusionMatrix(predictions, testResponse)




















### Linear Models

### Logistic regression
ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)

lr <- train(
  trainPredictors,
  y = trainresponse,
  method = "glm",
  metric = "ROC",
  trControl = ctrl
)


lr

predlr <- predict(lr,newdata=testPredictors)
testresponse <- factor(testresponse, levels = levels(predlr))
postResample(predlr,testresponse)
confusionMatrix(predlr,testresponse)

library(pROC)
prob_predictions <- predict(lr, testPredictors, type = "prob")[, 2]
roc_obj <- roc(testresponse, prob_predictions)
# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate
(Sensitivity)")
auc_value <- auc(roc_obj)
# Print the AUC value
print(auc_value)

## LDA
set.seed(489)
lda <- train(trainPredictors,
             y = trainresponse,
             method = "lda",
             metric = "ROC",
             trControl = ctrl,preProcess=c('center','scale'))
lda
plot(lda,main = "LDA plot")

# prediction
predlda <- predict(lda,newdata=testPredictors)
postResample(predlda,testresponse)

#confusion matrix
confusionMatrix(predlda,testresponse)

prob_predictions <- predict(lda, testPredictors, type = "prob")[, 2]
# Compute the ROC curve
library(pROC)
roc_obj <- roc(testresponse, prob_predictions)
# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate
(Sensitivity)")
auc_value <- auc(roc_obj)
# Print the AUC value
print(auc_value)

### PLSDA
set.seed(489)
plsda <- train(trainPredictors,
               y = trainresponse,
               method = "pls",
               metric = "ROC",
               tuneGrid = expand.grid(.ncomp = 1:10 ),
               trControl = ctrl,preProcess=c('center','scale'))
plsda
plot(plsda,main = "PLSDA plot")

#prediction
predplsda <- predict(plsda,newdata=testPredictors)
postResample(predplsda,testresponse)

# confusion
confusionMatrix(predplsda,testresponse)

prob_predictions <- predict(plsda, testPredictors, type = "prob")[, 2]
# Compute the ROC curve
library(pROC)
roc_obj <- roc(testresponse, prob_predictions)
# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate
(Sensitivity)")
auc_value <- auc(roc_obj)
# Print the AUC value
print(auc_value)

##Penalized Models
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)
glmnTuned <- train(trainPredictors,
                   trainresponse,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)
glmnTuned

plot(glmnTuned,main = "GLM plot ")

predglmn <- predict(glmnTuned,newdata=testPredictors)
postResample(predglmn,testresponse)

# confusion Matrix
confusionMatrix(predglmn,testresponse)

prob_predictions <- predict(glmnTuned, testPredictors, type = "prob")[, 2]
# Compute the ROC curve
library(pROC)
roc_obj <- roc(testresponse, prob_predictions)
# Plot the ROC curve
plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate
(Sensitivity)")
auc_value <- auc(roc_obj)
# Print the AUC value
print(auc_value)

################################################################################
#############  Non linear Models

### MDA
library(caret)
ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
) 

set.seed(476)
mdaFit <- train(trainPredictors, 
                trainresponse,
                method = "mda",
                metric = "Kappa",
                tuneGrid = expand.grid(.subclasses = 1:15),
                trControl = ctrl)
mdaFit
plot(mdaFit)
# prediction
predmda <- predict(mdaFit,newdata=testPredictors)
testresponse <- factor(testresponse, levels = levels(predmda))
postResample(predmda,testresponse)

# confusion Matrix
confusionMatrix(predmda,testresponse)

#prob_predictions <- predict(mdaFit, testPredictors, type = "prob")[, 2]
# Compute the ROC curve
#library(pROC)
#roc_obj <- roc(testresponse, prob_predictions)
# Plot the ROC curve
# plot(roc_obj, main = "ROC Curve", xlab = "False Positive Rate (1 - Specificity)", ylab = "True Positive Rate
# (Sensitivity)")
# auc_value <- auc(roc_obj)
# Print the AUC value
# print(auc_value)


#### RDA
#install.packages("rrcov")
ctrl <- trainControl(
  method = "repeatedcv",             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)
library(rrcov)
set.seed(476)
tunegrid <- expand.grid(
  .gamma = seq(0.1, 1, by = 0.1),    # Example values for gamma
  .lambda = seq(0.01, 0.1, by = 0.01) # Example values for lambda
)
rdaFit <- train(trainPredictors, 
                trainresponse,
                method = "rda",
                metric = "Kappa",
                tuneGrid = tunegrid,
                trControl = ctrl)
rdaFit
plot(rdaFit)
# prediction
predrda <- predict(rdaFit,newdata=testPredictors)
postResample(predrda,testresponse)

# confusion Matrix
confusionMatrix(predrda,testresponse)

###### QDA
library(caret)

# Set up the control function
ctrl <- trainControl(
  method = "repeatedcv",             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)

# Set a random seed for reproducibility
set.seed(476)

# Train the QDA model
qdaFit <- train(trainPredictors, 
                trainresponse, 
                method = "qda", 
                metric = "Kappa",
                trControl = ctrl)

# Print the model details
print(qdaFit)
plot(qdaFit)
# prediction
predqda <- predict(qdaFit,newdata=testPredictors)
postResample(predqda,testresponse)

# confusion Matrix
confusionMatrix(predqda,testresponse)

#########################################################
######## Neural Networks
nnetGrid <- expand.grid(.size = 1:40, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (11 + 1) + (maxSize+1)*2) 

set.seed(476)
library(caret)
ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)

nnetFit <- train(trainPredictors, 
                 trainresponse,
                 method = "nnet",
                 metric = "Kappa",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)
# prediction
prednnet <- predict(nnetFit,newdata=testPredictors)
postResample(prednnet,testresponse)

## confusion matrix
confusionMatrix(prednnet,testresponse)

#### FDA
set.seed(476)

ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)

library(earth)
library(Formula)
library(plotmo)
library(plotrix)
library(TeachingDemos)

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:50)

fdaTuned <- train(trainPredictors, 
                  trainresponse,
                  method = "fda",
                  metric = 'Kappa',
                  tuneGrid = marsGrid,
                  trControl = ctrl)
fdaTuned

plot(fdaTuned)
# prediction
predfda <- predict(fdaTuned,newdata=testPredictors)
postResample(predfda,testresponse)

## confusion matrix
confusionMatrix(predfda,testresponse)


### SVM
set.seed(476)
library(kernlab)
library(caret)
ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)
sigmaRangeReduced <- sigest(as.matrix(trainPredictors))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 8)))
svmRModel <- train(trainPredictors, 
                   trainresponse,
                   method = "svmRadial",
                   metric = "Kappa",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)

# prediction
predsvm <- predict(svmRModel,newdata=testPredictors)
postResample(predsvm,testresponse)

## confusion matrix
confusionMatrix(predsvm,testresponse)


### KNN
set.seed(476)
ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)

library(caret)

knnFit <- train(trainPredictors, 
                trainresponse,
                method = "knn",
                metric = "Kappa",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(.k = 1:20),
                trControl = ctrl)

knnFit
plot(knnFit)

# prediction
predknn <- predict(knnFit,newdata=testPredictors)
postResample(predknn,testresponse)

## confusion matrix
confusionMatrix(predknn,testresponse)


### Naive Bayes
library(caret)
ctrl <- trainControl(
  method = "repeatedcv" ,             
  number = 5,   
  repeats = 3,
  summaryFunction = defaultSummary,
  classProbs = TRUE,        
  savePredictions = TRUE
)
#install.packages("klaR")
library(klaR)
set.seed(476)
nbFit <- train( trainPredictors, 
                trainresponse,
                method = "nb",
                metric = "Kappa",
                #preProc = c("center", "scale"),
                tuneGrid = data.frame(.fL = 10,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)

nbFit
plot(nbFit)

# prediction
prednbFit <- predict(nbFit,newdata=testPredictors)
postResample(prednbFit,testresponse)

## confusion matrix
confusionMatrix(prednbFit,testresponse)

