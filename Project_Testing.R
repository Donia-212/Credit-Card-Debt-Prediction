library(tidyverse)
library(ggplot2)
library(gplots)
library(dplyr)
library(DAAG)
library(glmnet)
library(MASS)
library(car)
library(caret)
library(pls)
library(boot)
library(modelr)
library(Metrics)
library(broom)
library(AMR)
library(olsrr)
library(ggcorrplot)
library(rstanarm)
library(earth)
library(ggpubr)
library(gam)
library(splines)
library(stats)

rm(list=ls())

setwd('C:/Users/d2o0n/OneDrive/Documents/Course Materials/L2 S2/MDA/Project')
set.seed(8885)

train_data <-  as_tibble(read.csv("Train_Features.csv"))
train_label <-  as_tibble(read.csv("Train_Output.csv"))

data <- full_join(train_data, train_label, by = 'Index')
summary(data)

ggplot(data, aes(x = Limit, y = Balance)) + 
  geom_point() + 
  facet_wrap(~ Region + Student)

ggplot(data, aes(x = Rating, y = Balance)) + 
  geom_point() + 
  facet_wrap(~ Region + Student)

ggplot(data, aes(x = Income, y = Balance)) + 
  geom_point() + 
  facet_wrap(~ Region)

ggplot(data, aes(x = Married ,y = Balance, color = Region)) +
  geom_boxplot(outlier.colour = "orange", outlier.size = 2)    # 98 year old balance 1999

ggplot(data, aes(x = Student ,y = Balance, color = Region)) +
  geom_boxplot(outlier.colour = "orange", outlier.size = 2) 

# ---------------------------------------------------------------------------------------------------------

data$Own <- ifelse(data$Own == "Yes",1,0)
data$Student <- ifelse(data$Student == "Yes",1,0)
data$Married <- ifelse(data$Married == "Yes",1,0)

data <- data %>%
  mutate(Region = case_when(
    Region == "South" ~ 1,
    Region == "East"  ~ 2,
    Region == "West"  ~ 3
  ))

data <- data[-1]

data_split <- resample_partition(data, c(test = 0.3, train = 0.7))
data_train <- as_tibble(data_split$train)
data_test <- as_tibble(data_split$test)

set.seed(8885)


# ---------------------------------------------------------------------------------------------------------

# Check for multicollinearity:

dev.off()
data_2 <- data_train %>%dplyr::select(-Balance)
corr_matrix <-  round(cor(data_2), 2)
ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower",lab = TRUE) 
model <- lm(data = data_train, Balance ~ .)

# A high rating qualify for a higher credit limit
vif(model)



# ---------------------------------------------------------------------------------------------------------

# Fitting a linear regression model is not helpful as we have multicollinearity

step <- stepAIC(model, direction = 'both')
summary(step)


# Keep Limit, Income, Student, Limit, Cards, Rating
# Remove own, education, married, region, age
# Best linear model: Balance ~ Income + Limit + Rating + Cards + Student

# Post-processing: Transform the predictions (nonnegative): 
non_negative_model <- function(m){
  for(i in seq(1,length(m))){
    if(m[i]< 0){
      m[i] = 0
    }
  }
  return(as_tibble(m))
}

y <- data_train$Balance
x <- as.matrix(data_train[c('Income','Limit','Rating','Cards','Age','Student')])
x_test <- as.matrix(data_test[c('Income','Limit','Rating','Cards','Age','Student')])

# model_mlr <- lm(data = data_train, Balance ~ .)
# stepAIC(model_mlr, direction = 'both')


model_mlr <- lm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student)
y_mlr <- predict(model_mlr, data_test)

SSE <- sum((non_negative_model(y_mlr) - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST    # 0.9776657



# ---------------------------------------------------------------------------------------------------------

# Ridge Regression: Model 1 is the best


# Selecting Features:
model <- cv.glmnet(x,y,alpha = 0)   
best_lambda <- model$lambda.min  
model_ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda)

y_ridge <- predict(model_ridge, x_test)
SSE <- sum((non_negative_model(y_ridge) - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST # 0.9493181


# Full Model:
# x_2 <- as.matrix(data_train[1:10])
# x_test_2 <- as.matrix(data_test[1:10])
# model_2 <- cv.glmnet(x_2,y,alpha = 0)   
# best_lambda_2 <- model_2$lambda.min  
# model_ridge_2 <- glmnet(x_2, y, alpha = 0, lambda = best_lambda_2)
# 
# y_ridge_2 <- predict(model_ridge_2, x_test_2)
# SSE <- sum((non_negative_model(y_ridge_2) - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST    # 0.9482224
# 
# 
# # Omitting Limit:
# x_3 <- as.matrix(data_train[c('Income', 'Rating', 'Cards', 'Age', 'Education', 'Own', 'Student', 'Married', 'Region')])
# 
# model_3 <- cv.glmnet(x_3,y,alpha = 0)   
# best_lambda_3 <- model_3$lambda.min  
# model_ridge_3 <- glmnet(x_3, y, alpha = 0, lambda = best_lambda_3)
# c_ridge_3 <- coefficients(model_ridge_3) 
# 
# y_ridge_3 <- c_ridge_3[1] + c_ridge_3[2]*data_test$Income + c_ridge_3[3]*data_test$Rating + c_ridge_3[4]*data_test$Cards + c_ridge_3[5]*data_test$Age + c_ridge_3[6]*data_test$Education + c_ridge_3[7]*data_test$Own + c_ridge_3[8]*data_test$Student + c_ridge_3[9]*data_test$Married + c_ridge_3[10]*data_test$Region
# y_ridge_3 <- non_negative_model(y_ridge_3)
# SSE <- sum((y_ridge_3 - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST    # 0.9184196



# ---------------------------------------------------------------------------------------------------------

# LASSO Regression: Model 1 is the best

model_4 <- cv.glmnet(x, y, alpha = 1)
best_lambda_4 <- model_4$lambda.min
model_LASSO <- glmnet(x, y, alpha = 1, lambda = best_lambda_4)
c_LASSO <- coef(model_LASSO)

y_LASSO <- predict(model_LASSO, x_test)
SSE <- sum((non_negative_model(y_LASSO) - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.977186


# model_5 <- cv.glmnet(x_2, y, alpha = 1)
# best_lambda_5 <- model_5$lambda.min
# model_LASSO_2 <- glmnet(x_2, y, alpha = 1, lambda = best_lambda_5)
# c_LASSO_2 <- coef(model_LASSO_2)
# 
# y_LASSO_2 <- c_LASSO_2[1] + c_LASSO_2[2]*data_test$Income + c_LASSO_2[3]*data_test$Limit + c_LASSO_2[4]*data_test$Rating + c_LASSO_2[5]*data_test$Cards + + c_LASSO_2[6]*data_test$Age + c_LASSO_2[7]*data_test$Education + c_LASSO_2[8]*data_test$Own + c_LASSO_2[9]*data_test$Student + c_LASSO_2[10]*data_test$Married + c_LASSO_2[11]*data_test$Region 
# y_LASSO_2 <- non_negative_model(y_LASSO_2)
# SSE <- sum((y_LASSO_2 - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST  # 0.9753854



# ---------------------------------------------------------------------------------------------------------

# Elastic Net Regression: Model 1 is the best

model_6 <- cv.glmnet(x, y, alpha = .5)
best_lambda_6 <- model_6$lambda.min
model_EN <- glmnet(x, y, alpha = .5, lambda = best_lambda_6)

y_EN <- predict(model_EN, x_test)
SSE <- sum((non_negative_model(y_EN) - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9773034


# model_7 <- cv.glmnet(x_2, y, alpha = .5)
# best_lambda_7 <- model_7$lambda.min
# model_EN_2 <- glmnet(x_2, y, alpha = .5, lambda = best_lambda_7)
# c_EN_2 <- coef(model_EN_2)
# 
# y_EN_2 <- c_EN_2[1] + c_EN_2[2]*data_test$Income + c_EN_2[3]*data_test$Limit + c_EN_2[4]*data_test$Rating + c_EN_2[5]*data_test$Cards + c_EN_2[6]*data_test$Age + c_EN_2[7]*data_test$Education + c_EN_2[8]*data_test$Own + c_EN_2[9]*data_test$Student + c_EN_2[10]*data_test$Married + c_EN_2[11]*data_test$Region 
# y_EN_2 <- non_negative_model(y_EN_2)
# SSE <- sum((y_EN_2 - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST  # 0.9749697



# ---------------------------------------------------------------------------------------------------------

# Bayesian Linear Regression: Model 1

set.seed(8885)

model_bayesian <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student)
c_b <- coefficients(model_bayesian) 
y_b <- c_b[1] + c_b[2]*data_test$Income + c_b[3]*data_test$Limit + c_b[4]*data_test$Rating + c_b[5]*data_test$Cards + c_b[6]*data_test$Age + c_b[7]*data_test$Student
non_negative_y_b <- non_negative_model(y_b)
SSE <- sum((non_negative_y_b - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST    # 0.9776458 

set.seed(8885)

# model_bayesian <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards +Student)
# c_b <- coefficients(model_bayesian) 
# y_b <- c_b[1] + c_b[2]*data_test$Income + c_b[3]*data_test$Limit + c_b[4]*data_test$Rating + c_b[5]*data_test$Cards + c_b[6]*data_test$Student
# non_negative_model_y_b <- non_negative_model(y_b)
# SSE <- sum((non_negative_model_y_b - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST   # 0.9772653
# 
# model_bayesian_2 <- stan_glm(data = data_train, Balance ~ .)
# c_b_2 <- model_bayesian_2$coefficients
# 
# y_b_2 <- c_b_2[1] + c_b_2[2]*data_test$Income + c_b_2[3]*data_test$Limit + c_b_2[4]*data_test$Rating + c_b_2[5]*data_test$Cards + c_b_2[6]*data_test$Age + c_b_2[7]*data_test$Education + c_b_2[8]*data_test$Own + c_b_2[9]*data_test$Student + c_b_2[10]*data_test$Married + c_b_2[11]*data_test$Region
# non_negative_model_y_b_2 <- non_negative_model(y_b_2)
# SSE <- sum((non_negative_model_y_b_2 - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST    # 0.9755138



# ---------------------------------------------------------------------------------------------------------

# Model Inform Model: Model 3 is the best

y_mlr_pred <- predict(model_mlr, newx = x)
y_ridge_pred <- predict(model_ridge, newx = x)
y_LASSO_pred <- predict(model_LASSO, newx = x)
y_EN_pred <- predict(model_EN, newx = x)
y_B_pred <- predict(model_bayesian, newx = x)

y_ridge_test <- predict(model_ridge, newx = x_test)
y_LASSO_pred <- predict(model_LASSO, newx = x)

data_3 <- data_train %>%
  mutate(MLR = model_mlr$fitted.values, Ridge = y_ridge_pred, LASSO = y_LASSO_pred, EN = model_EN$fitted.values, B = model_bayesian$fitted.values )

data_test_2 <- data_test %>%
  mutate(MLR = y_mlr, Ridge = y_ridge, LASSO = y_LASSO, EN = y_EN, B = y_b )


# The model is significant but the coefficients are misleading
model_inform <- lm(data = data_3, Balance ~ MLR + Ridge + LASSO) 
anova(model_inform)
summary(model_inform)   
y_inform <- add_predictions(model_inform, data = data_test_2)$pred
SSE <- sum((non_negative_model(y_inform) - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9776657

# Almost and MLR model the other coefficients are almost zero
model_inform_2 <- lm(data = data_3, Balance ~ Ridge + LASSO) 
anova(model_inform_2)     # The coefficients are significant
summary(model_inform_2)   # The model is significant
c_inform_2 <- model_inform_2$coefficients

y_inform_2 <- c_inform_2[1] + c_inform_2[2] * y_ridge + c_inform_2[3] * y_LASSO
non_negative_y_inform_2 <- non_negative_model(y_inform_2)
SSE <- sum((non_negative_y_inform_2 - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9778481


# model_inform_3 <- lm(data = data_3, Balance ~ EN + B) 
# anova(model_inform_3)     # The coefficients of EN & B might be misleading
# summary(model_inform_3)   # The model is significant
# c_inform_3 <- model_inform_3$coefficients
# 
# y_inform_3 <- c_inform_3[1] + c_inform_3[2]* y_EN + c_inform_3[3] * y_b
# non_negative_y_inform_3 <- non_negative_model(y_inform_3)
# SSE <- sum((non_negative_y_inform_3 - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST  # 0.9776215




# ---------------------------------------------------------------------------------------------------------

# Polynomial Regression:

model_poly <- lm(data = data_train, Balance ~ Income+Rating+Limit+Cards+Age+Student+I(Income^2)+I(Rating^2)+Student:Limit+Student:Rating+Student:Cards+Student:Age+Income:Student+Limit:Rating:Age+Limit:Rating:Student+Income:Rating:Limit+Income:Student:Age+Income:Rating:Limit:Student:Cards:Age) 
c_poly <- model_poly$coefficients
stepAIC(model_poly, direction = 'both')

y_poly <- c_poly[1] + c_poly[2] * data_test$Income + c_poly[3] * data_test$Rating + c_poly[4] * data_test$Limit + c_poly[5] * data_test$Cards + c_poly[6] * data_test$Age + c_poly[7] * data_test$Student + c_poly[8] * (data_test$Income)^2 + c_poly[9] * (data_test$Rating)^2 +  c_poly[10] * data_test$Limit * data_test$Student + c_poly[11] * data_test$Rating * data_test$Student +  c_poly[12] * data_test$Cards * data_test$Student + c_poly[13] * data_test$Age * data_test$Student +  c_poly[14] * data_test$Income * data_test$Student +  c_poly[15] * data_test$Rating * data_test$Age * data_test$Limit + c_poly[16] * data_test$Student * data_test$Rating * data_test$Limit + c_poly[17] * data_test$Income * data_test$Rating * data_test$Limit +  c_poly[18] * data_test$Income * data_test$Age * data_test$Student + c_poly[19] * data_test$Income * data_test$Rating * data_test$Limit * data_test$Student * data_test$Age * data_test$Cards
non_negative_y_poly <- non_negative_model(y_poly)
SSE <- sum((non_negative_y_poly - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9888849



# ---------------------------------------------------------------------------------------------------------

# Log Regression:
# 
# data_train_2 <- data_train %>% mutate(Student = case_when(
#   Student == 1 ~ 1,
#   Student == 0 ~ 2
# ))
# 
# data_test_2 <- data_test %>% mutate(Student = case_when(
#   Student == 1 ~ 1,
#   Student == 0 ~ 2
# ))
# 
# model_log <- lm(data = data_train_2, Balance ~ log(Income) + log(Rating) + log(Limit) + log(Cards) + log(Student))
# c_log <- model_log$coefficients
# 
# y_log <- c_log[1] + c_log[2] * log(data_test_2$Income) + c_log[3] * log(data_test_2$Rating) + c_log[4] * log(data_test_2$Limit) + c_log[5] * log(data_test_2$Cards) + c_log[6] * log(data_test_2$Student)
# non_negative_y_log <- non_negative_model(y_log)
# SSE <- sum((non_negative_y_log - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST  # 0.8556709



# ---------------------------------------------------------------------------------------------------------

# Non Linear Model Inform Model: Not a good model


# model_poly_inform <- lm(data = data_3, Balance ~ MLR + EN + B + I(MLR^2) + I(EN^2) + I(B^2) + MLR:EN + EN:B + MLR:EN:B) 
# c_poly_inform <- model_poly_inform$coefficients
# 
# y_poly_inform <- c_poly_inform[1] + c_poly_inform[2] * y_mlr + c_poly_inform[3] * y_EN + c_poly_inform[4] * y_b + c_poly_inform[5] * (y_mlr)^2 + c_poly_inform[6] * (y_EN)^2 + c_poly_inform[7] * (y_b)^2 + c_poly_inform[8] * y_mlr * y_EN +  c_poly_inform[9] * y_EN * y_b +  c_poly_inform[10] * y_mlr * y_EN * y_b
# non_negative_y_poly_inform <- non_negative_model(y_poly_inform)
# SSE <- sum((non_negative_y_poly_inform - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST  # 0.9949548

# summary(stepAIC(model_poly_inform, direction = 'both'))

model_poly_inform <- lm(data = data_3, Balance ~ MLR + EN + B + I(MLR^2) + I(B^2) + MLR:EN + MLR:B + EN:B + MLR:EN:B) 
c_poly_inform <- model_poly_inform$coefficients

y_poly_inform <- c_poly_inform[1] + c_poly_inform[2] * y_mlr + c_poly_inform[3] * y_EN + c_poly_inform[4] * y_b + c_poly_inform[5] * (y_mlr)^2 + c_poly_inform[6] * (y_b)^2 + c_poly_inform[7] * y_mlr * y_EN +  c_poly_inform[8] * y_mlr * y_b + c_poly_inform[9] * y_EN * y_b +  c_poly_inform[10] * y_mlr * y_EN * y_b
non_negative_y_poly_inform <- non_negative_model(y_poly_inform)
SSE <- sum((non_negative_y_poly_inform - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9950379



# ---------------------------------------------------------------------------------------------------------



# LASSO Inform Model: 

y_Poly_pred <- predict(model_poly, newx = x)
y_Poly_inform_pred <- predict(model_poly_inform, newx = x)

data_3 <- data_3 %>% mutate(Poly = y_Poly_pred, Poly_Inform = y_Poly_inform_pred)

x_6 <- as.matrix(data_3[c('MLR','EN','B','Poly','Poly_Inform')])

model_9 <- cv.glmnet(x_6,y, alpha = 1)
best_lambda_9 <- model_9$lambda.min
model_LASSO_inform <- glmnet(x_6, y, alpha = 1, lambda = best_lambda_9)
c_LASSO_inform <- coef(model_LASSO_inform)

y_LASSO_inform <- c_LASSO_inform[1] + c_LASSO_inform[2] * y_mlr + c_LASSO_inform[3] * y_EN + c_LASSO_inform[4] * y_b + c_LASSO_inform[5] * y_poly + c_LASSO_inform[6] * y_poly_inform
non_negative_y_LASSO_inform <- non_negative_model(y_LASSO_inform)
SSE <- sum((non_negative_y_LASSO_inform - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9951148



# EN Inform Model: 

model_10 <- cv.glmnet(x_6,y, alpha = .5)
best_lambda_10 <- model_10$lambda.min
model_EN_inform <- glmnet(x_6, y, alpha = .5, lambda = best_lambda_10)
c_EN_inform <- coef(model_EN_inform)

y_EN_inform <- c_EN_inform[1] + c_EN_inform[2] * y_mlr + c_EN_inform[3] * y_EN + c_EN_inform[4] * y_b + c_EN_inform[5] * y_poly + c_EN_inform[6] * y_poly_inform
non_negative_y_EN_inform <- non_negative_model(y_EN_inform)
SSE <- sum((non_negative_y_EN_inform - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9958321


# Ridge Inform Model: 

x_7 <- as.matrix(data_3[c('MLR','EN','B')])
model_8 <- cv.glmnet(x_7,y, alpha = 0)
best_lambda_8 <- model_8$lambda.min
model_ridge_inform <- glmnet(x_7, y, alpha = 0, lambda = best_lambda_8)
c_ridge_inform <- coef(model_ridge_inform)

y_ridge_inform <- c_ridge_inform[1] + c_ridge_inform[2] * y_mlr + c_ridge_inform[3] * y_EN + c_ridge_inform[4] * y_b
non_negative_y_ridge_inform <- non_negative_model(y_ridge_inform)
SSE <- sum((non_negative_y_ridge_inform - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9718435



# Ridge Inform:

y_LASSO_inform_pred <- predict(model_LASSO_inform, newx = x_6)
y_EN_inform_pred <- predict(model_EN_inform, newx = x_6)
y_Ridge_inform_pred <- predict(model_ridge_inform, newx = x_7)

data_3 <- data_3 %>% mutate(LASSO_Inform = y_LASSO_inform_pred, EN_Inform = y_EN_inform_pred, Ridge_Inform = y_Ridge_inform_pred)

x_8 <- as.matrix(data_3[c('Poly_Inform', 'LASSO_Inform', 'EN_Inform')])

model_9 <- cv.glmnet(x_8,y, alpha = 0)
best_lambda_9 <- model_9$lambda.min
model_ridge_inform_2 <- glmnet(x_8, y, alpha = 0, lambda = best_lambda_9)
c_ridge_inform_2 <- coef(model_ridge_inform_2)

y_ridge_inform_2 <- c_ridge_inform_2[1] + c_ridge_inform_2[2] * y_poly_inform + c_ridge_inform_2[3] * y_LASSO_inform + c_ridge_inform[4] * y_EN_inform 
non_negative_y_ridge_inform_2 <- non_negative_model(y_ridge_inform_2)
SSE <- sum((non_negative_y_ridge_inform_2 - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST   # 0.9951935

y_Ridge_inform_2_pred <- predict(model_ridge_inform_2, newx = x_8)
data_4 <- data_3 %>% mutate(Ridge_INFORM_2 = y_Ridge_inform_2_pred)



# Poly Inform 2: 
model_poly_inform_2 <- lm(data = data_4, Balance ~ Poly_Inform + EN_Inform + I(LASSO_Inform^2) + I(EN_Inform^2) + I(Ridge_INFORM_2^2) + Poly_Inform:EN_Inform:Ridge_INFORM_2 + Poly_Inform:EN_Inform:LASSO_Inform) 
c_poly_inform_2 <- model_poly_inform_2$coefficients

y_poly_inform_2 <- c_poly_inform_2[1] + c_poly_inform_2[2] * y_poly_inform + c_poly_inform_2[3] * y_EN_inform + c_poly_inform_2[4] * (y_LASSO_inform)^2 + c_poly_inform_2[5] * (y_EN_inform)^2 + c_poly_inform_2[6] * (y_ridge_inform_2)^2 + c_poly_inform_2[7] * y_poly_inform * y_EN_inform * y_ridge_inform_2 +  c_poly_inform_2[8] * y_poly_inform * y_EN_inform * y_LASSO_inform

non_negative_y_poly_inform_2 <- non_negative_model(y_poly_inform_2)
SSE <- sum((non_negative_y_poly_inform_2 - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST 




# Model: MARS

# Create a parameter tuning 'grid
parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))

# Cross Validation
cv_mars_model <- train(x = x, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
ggplot(cv_mars_model)
y_mars <- predict(object = cv_mars_model$finalModel, newdata = data_test) 

non_negative_y_mars <- non_negative_model(y_mars)
SSE <- sum((non_negative_y_mars - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST  # 0.9970157


# Create a parameter tuning 'grid
# parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))
# 
# # Cross Validation
# cv_mars_model <- train(x = x_2, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
# ggplot(cv_mars_model)
# y_mars <- predict(object = cv_mars_model$finalModel, newdata = data_test) 
# 
# non_negative_y_mars <- non_negative_model(y_mars)
# SSE <- sum((non_negative_y_mars - data_test$Balance)^2)
# SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
# 1- SSE/SST


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Model: Elastic Net

x_2 <- as.matrix(data_train[1:10])
x_test <- as.matrix(data_test[c('Income','Limit','Rating','Cards','Age','Student')])

parameter_grid_en <- expand.grid(alpha = seq(0,1, length = 10), lambda = seq(0.0001,0.2, length = 5))

train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              verboseIter = TRUE)

cv_en<- train(x,y, method = "glmnet", metric = 'Rsquared', trControl = train_control, tuneGrid = parameter_grid_en)
model_EN <- glmnet(x, y, alpha = cv_en$bestTune[1], lambda = cv_en$bestTune[2])
y_en <- predict(model_EN, x_test)

non_negative_y_en <- non_negative_model(y_en)
SSE <- sum((non_negative_y_en - data_test$Balance)^2)
SST <- sum((mean(data_test$Balance) - data_test$Balance)^2)
1- SSE/SST   # 0.9776163