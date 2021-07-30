

################################################################################
# install packages if necessary
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(corrplot)) install.packages("corrplot")
if(!require(Rborist)) install.packages("Rborist")
if(!require(randomForest)) install.packages("randomForest")
if(!require(e1071)) install.packages("e1071")
#if(!require(merTools)) install.packages("merTools")


# load libraries
library(tidyverse)
library(lubridate)
#library(GGally)
library(caret)
library(Rborist)
library(randomForest)
library(e1071)
library(corrplot)
library(knitr)
#library(merTools)

################################################################################

# set seed
set.seed(2021)
options(digits = 5)

# load csv file with data

current_path <- getwd()
file_name <- "creditcard.csv"

full_path_creditcard <- file.path(current_path, file_name)

credit_card_data <- read_csv(full_path_creditcard)

# change class to factored data
credit_card_data_factored <- credit_card_data %>% mutate(Class=factor(Class))

################################################################################

# check for na records in the dataset to ensure it's clean
summary_na_vals <- sapply(credit_card_data, function(x){
  sum(is.na(x))
})


# correlation between values in the database
cor(credit_card_data)



# count fraudulent transaction. Note that a 1 represents a fraudulent transaction.
# this shows that the data is imbalanced significantly. Fraudulent transaction represents less than 1% of the data. 
credit_card_data %>% group_by(Class) %>% summarise(counts = n()) %>% summarise(Class=Class, counts=counts, proportions=counts/sum(counts))

################################################################################

# Visual analysis

# correlation matrix to understand the relationship between pairs of values.
corrplot(cor(credit_card_data[,1:30]), tl.cex = 0.5, tl.offset = 0.6)


# amounts on fraudulent transactions
credit_card_data_factored %>% filter(Class==1) %>% ggplot(aes(Amount)) + #geom_histogram(binwidth = 50)
  scale_x_continuous(trans = "sqrt", oob = scales::squish_infinite) + geom_histogram(binwidth = 2)

# amounts on non-fraudulent transactions
credit_card_data_factored %>% filter(Class==0) %>% ggplot(aes(Amount)) +
  scale_x_continuous(trans = "log2", oob = scales::squish_infinite) + geom_histogram(binwidth = 3)



# time in seconds verses the amount of each transaction and then the classes in color
credit_card_data_factored %>% ggplot(aes(Amount, Time, color=Class))  + 
  scale_x_continuous(trans = "sqrt", oob = scales::squish_infinite) + geom_point()




################################################################################

# summary metrics
# mean of all transactions
mean(credit_card_data$Amount)

#mean of fraud trans
credit_card_data %>% filter(Class==0) %>% summarise(mean_0 = mean(Amount), sd_0 = sd(Amount), max_0 = max(Amount), min_0 = min(Amount))

# means non-fraud trans
# shows that the mean of fraud transactions is higher overall.
credit_card_data %>% filter(Class==1) %>% summarise(mean_1 = mean(Amount), sd_1 = sd(Amount), max_1 = max(Amount), min_1 = min(Amount))


# shows that the fraud data points are not that different when plotted against time. 
credit_card_data %>% ggplot(aes(Amount, Time, color=Class)) + geom_point()


################################################################################

# sub sample the non-fraud transactions
fraud_class <- credit_card_data_factored %>% filter(Class==1)
non_fraud_class <- credit_card_data_factored %>% filter(Class==0)
sample_non_fraud <- sample_n(non_fraud_class, 120000, replace = FALSE)
merge_fraud_non_fraud <- rbind(fraud_class, sample_non_fraud)
final_dataset <- sample_n(merge_fraud_non_fraud, nrow(merge_fraud_non_fraud), replace = FALSE)

# scale the dataset
scaled_final_dataset <- as.data.frame(scale(final_dataset[, 1:30]))
scaled_final_dataset <- cbind(scaled_final_dataset, final_dataset[,31])

# split data into training and validation sets

test_indexes <- createDataPartition(scaled_final_dataset$Class, times=1, p=0.2, list=FALSE)

# create validation and training sets.
validation_set <- scaled_final_dataset[test_indexes,]
training_set <- scaled_final_dataset[-test_indexes,]


# clean up the datasets not used, only keep the validation and training datasets we are using for modeling
# rm(fraud_class, non_fraud_class, sample_non_fraud, merge_fraud_non_fraud, credit_card_data_factored, credit_card_data)

################################################################################

train_params <- trainControl(method = 'cv', number = 3, p=0.9)

# GLM model with no - class weights
glm_fit <- train(Class ~ ., data=training_set, method='glm', trControl=train_params, family=quasibinomial, maxit=100)
glm_predictions <- predict(glm_fit, newdata=validation_set)
glm_cm <- confusionMatrix(glm_predictions, factor(validation_set$Class))

# table of results
model_results <- tibble(Method = "Logistic Regression", Sensitivity=glm_cm$byClass[1], Specificity=glm_cm$byClass[2], Accuracy=glm_cm$overall[1])


# confusion matrix plot glm model
glm_cm_tb <- as.data.frame(glm_cm$table)
glm_cm_tb %>% ggplot(aes(Prediction, Reference)) + geom_tile(aes(fill=Freq)) +
  geom_text(aes(label=Freq)) +
  scale_fill_gradient2() +
  ggtitle("Logistic Regression - Confusion Matrix")
  



# Use class weights given the data is imbalanced
weights_0_1 <- training_set %>% group_by(Class) %>% summarise(counts = n()) %>% mutate(weight = (1/counts)*0.5) %>% pull(weight)
# create vector of training weights
training_weights <- training_set %>% summarise(weight = ifelse(Class == 0, weights_0_1[1], weights_0_1[2])) %>% pull(weight)


# GLM model with class weights
glm_fit_cl_weight <- train(Class ~ ., data=training_set, method='glm', 
                           trControl=train_params, weights = training_weights, family=quasibinomial, maxit=100)

glm_predictions_cl_weight <- predict(glm_fit_cl_weight, newdata=validation_set)
glm_cl_weight_cm <- confusionMatrix(glm_predictions_cl_weight, factor(validation_set$Class))


# update table of results.
model_results <- bind_rows(model_results, tibble(Method = "Logistic Regression - with class weights", 
                                                 Sensitivity=glm_cl_weight_cm$byClass[1], 
                                                 Specificity=glm_cl_weight_cm$byClass[2], 
                                                 Accuracy=glm_cl_weight_cm$overall[1]))

# confusion matrix plot glm model with class weights
glm_cm_cl_w_tb <- as.data.frame(glm_cl_weight_cm$table)
glm_cm_cl_w_tb %>% ggplot(aes(Prediction, Reference)) + geom_tile(aes(fill=Freq)) +
  geom_text(aes(label=Freq)) +
  scale_fill_gradient2(low = 'red', high = 'green') +
  ggtitle("Logistic Regression - Class Weights Confusion Matrix")



# SVM Model
svm_fit <- svm(Class ~ ., data=training_set, class.weights=c("0"=weights_0_1[1], "1"=weights_0_1[2]), kernel='linear')
svm_predictions <- predict(svm_fit, newdata=validation_set)
svm_cm <- confusionMatrix(svm_predictions, factor(validation_set$Class))

# update table of results.
model_results <- bind_rows(model_results, tibble(Method = "SVM - with class weights", 
                                                 Sensitivity=svm_cm$byClass[1], 
                                                 Specificity=svm_cm$byClass[2], 
                                                 Accuracy=svm_cm$overall[1]))

# confusion matrix plot svm model with class weights
svm_cm_tb <- as.data.frame(svm_cm$table)
svm_cm_tb %>% ggplot(aes(Prediction, Reference)) + geom_tile(aes(fill=Freq)) +
  geom_text(aes(label=Freq)) +
  scale_fill_gradient2(low = 'red', high = 'blue') +
  ggtitle("SVM - Confusion Matrix")


# random forest
# update the training and validation sets to not have feature scaling
# split data into training and validation sets

test_indexes <- createDataPartition(final_dataset$Class, times=1, p=0.2, list=FALSE)

# create validation and training sets.
validation_set_rf <- final_dataset[test_indexes,]
training_set_rf <- final_dataset[-test_indexes,]


# results of running the model showed an mtry value of 16 as optimal
rf_fit <- train(Class ~ ., data=training_set_rf, method='rf', 
                classwt=c("0"=weights_0_1[1], "1"=weights_0_1[2]), trControl=train_params)

rf_predictions <- predict(rf_fit, newdata = validation_set_rf)
rf_cm <- confusionMatrix(rf_predictions, factor(validation_set_rf$Class))

# update table of results.
model_results <- bind_rows(model_results, tibble(Method = "Random Forest - with class weights", 
                                                 Sensitivity=rf_cm$byClass[1], Specificity=rf_cm$byClass[2], 
                                                 Accuracy=rf_cm$overall[1]))

# confusion matrix plot random forest model with class weights
rf_cm_tb <- as.data.frame(rf_cm$table)
rf_cm_tb %>% ggplot(aes(Prediction, Reference)) + geom_tile(aes(fill=Freq)) +
  geom_text(aes(label=Freq)) +
  scale_fill_gradient2(low = 'yellow', high = 'purple') +
  ggtitle("Random Forest - Confusion Matrix")



# plot of Specificity and Sensitivity
model_results %>% ggplot(aes(Specificity, Sensitivity, color=Method)) + geom_point(size=5)

