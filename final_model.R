############################
# Setting up the Environment
############################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")

################
# Data Wrangling
################

# download and store csv file in dat
tmp=tempfile()
download.file("https://raw.githubusercontent.com/PRerk/Graduate-Admission-Prediction/master/datasets_14872_228180_Admission_Predict_Ver1.1.csv",tmp)
dat <- read_csv(tmp)
file.remove(tmp)

# set seed for testing
set.seed(1,sample.kind = "Rounding")

# create train set and test set from dat
test_index <- createDataPartition(y = dat$`Chance of Admit`, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- dat[-test_index,]
test_set <- dat[test_index,]

######################################
# Create function for calculating RMSE
######################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

###################
# Linear Regression
###################

# training with lm
train_lm <- train(`Chance of Admit`~.,
                  data = train_set,
                  method = "lm")
# predicting chance of admission
pred_lm <- predict(train_lm, test_set)
# calulating RMSE
rmse_lm <- RMSE(test_set$`Chance of Admit`,pred_lm)
# store RMSE in rmse_results
rmse_results <- tibble(method = "lm", RMSE = rmse_lm)

#####################
# k-nearest neighbors
#####################

# training with knn
train_knn <- train(`Chance of Admit`~.,
                  data = train_set,
                  method = "knn",
                  tuneGrid = data.frame(k = c(3,5,7,9,11)))
# predicting chance of admission
pred_knn <- predict(train_knn, test_set)
# calculating RMSE
rmse_knn <- RMSE(test_set$`Chance of Admit`,pred_knn)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="knn",
                                     RMSE = rmse_knn ))

################
# Random forests
################

# training with rf
train_rf <- train(`Chance of Admit`~.,
                   data = train_set,
                   method = "rf")
# predicting chance of admission
pred_rf <- predict(train_rf, test_set)
# calculating RMSE
rmse_rf <- RMSE(test_set$`Chance of Admit`,pred_rf)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="rf",
                                 RMSE = rmse_rf ))

###########################
# Artificial neural network
###########################

# training with nnet
train_nnet <- train(`Chance of Admit`~.,
                    data = train_set,
                    method = "nnet")
# predicting chance of admission
pred_nnet <- predict(train_nnet, test_set)
# calculating RMSE
rmse_nnet <- RMSE(test_set$`Chance of Admit`,pred_nnet)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="nnet",
                                 RMSE = rmse_nnet ))

##########################################################
# Ensemble: Ensemble: Linear Regression and Random Forests
##########################################################

# predicting chance of admission using ensemble model
pred_en <- (pred_lm+pred_rf)/2
# calculating RMSE
rmse_en <- RMSE(test_set$`Chance of Admit`,pred_en)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="ensemble: lm + rf",
                                 RMSE = rmse_en ))

########################
# Gradient Boosted Model
########################

# training with gbm
train_gbm <- train(`Chance of Admit`~.,
                    data = train_set,
                    method = "gbm")
# predicting chance of admission
pred_gbm <- predict(train_nnet, test_set)
# calculating RMSE
rmse_gbm <- RMSE(test_set$`Chance of Admit`,pred_gbm)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="gbm",
                                 RMSE = rmse_gbm ))

########################
# Support Vector Machine
########################

# training with svm linear
train_svm <- train(`Chance of Admit`~.,
                   data = train_set,
                   method = "svmLinear")

# predicting chance of admission
pred_svm <- predict(train_svm, test_set)
# calculating RMSE
rmse_svm <- RMSE(test_set$`Chance of Admit`,pred_svm)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="svm",
                                 RMSE = rmse_svm ))

# training with svm non-linear: radial
train_svm_radial <- train(`Chance of Admit`~.,
                   data = train_set,
                   method = "svmRadial")
# predicting chance of admission
pred_svm_radial <- predict(train_svm_radial, test_set)
# calculating RMSE
rmse_svm_radial <- RMSE(test_set$`Chance of Admit`,pred_svm_radial)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="svm non linear - Radial",
                                 RMSE = rmse_svm_radial ))

# training with svm non-linear: poly
train_svm_poly <- train(`Chance of Admit`~.,
                          data = train_set,
                          method = "svmPoly")
# predicting chance of admission
pred_svm_poly <- predict(train_svm_poly, test_set)
# calculating RMSE
rmse_svm_poly <- RMSE(test_set$`Chance of Admit`,pred_svm_poly)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="svm non linear - Poly",
                                 RMSE = rmse_svm_poly ))

#########################################
# Ensemble: Random Forests and SVM Radial
#########################################

# predicting chance of admission using new ensemble model
pred_en2 <- (pred_rf+pred_svm_radial)/2
# calculating RMSE
rmse_en2 <- RMSE(test_set$`Chance of Admit`,pred_en2)
# store RMSE in rmse_results
rmse_results <- bind_rows(rmse_results,
                          tibble(method="ensemble rf + svm radial",
                                 RMSE = rmse_en2 ))

#################################
# Displaying result of the models
#################################
rmse_results
