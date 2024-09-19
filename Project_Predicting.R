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

data_test <-  as_tibble(read.csv("Test_Features.csv"))
data_train <-  as_tibble(read.csv("Train_Features.csv"))
train_label <-  as_tibble(read.csv("Train_Output.csv"))

data_train <- full_join(data_train, train_label, by = 'Index')

data_train$Own <- ifelse(data_train$Own == "Yes",1,0)
data_train$Student <- ifelse(data_train$Student == "Yes",1,0)
data_train$Married <- ifelse(data_train$Married == "Yes",1,0)

data_train <- data_train %>%
  mutate(Region = case_when(
    Region == "South" ~ 1,
    Region == "East"  ~ 2,
    Region == "West"  ~ 3
  ))


data_train <- data_train[-1]


# We will use the best models in Project_Testing.R and apply them to the full data to predict the Balance
# of data_test and submit the results

non_negative_model <- function(m){
  for(i in seq(1,length(m))){
    if(m[i]< 0){
      m[i] = 0
    }
  }
  return(as_tibble(m))
}

# Model 1: MLR

model_mlr <- lm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student)
c_mlr <- coefficients(model_mlr) 
y_mlr <- c_mlr[1] + c_mlr[2]*data_test$Income + c_mlr[3]*data_test$Limit + c_mlr[4]*data_test$Rating + c_mlr[5]*data_test$Cards + c_mlr[6]*data_test$Age + c_mlr[7]*data_test$Student
#View(non_negative_model(y_mlr))



# Model 2: Ridge

x <- as.matrix(data_train[c('Income','Limit','Rating','Cards','Age','Student')])
y <- data_train$Balance

model <- cv.glmnet(x,y,alpha = 0)   
best_lambda <- model$lambda.min  
model_ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda)
c_ridge <- coefficients(model_ridge) 
y_ridge <- c_ridge[1] + c_ridge[2]*data_test$Income + c_ridge[3]*data_test$Limit + c_ridge[4]*data_test$Rating + c_ridge[5]*data_test$Cards + c_ridge[6]*data_test$Age + c_ridge[7]*data_test$Student
#View(non_negative_model(y_ridge))



# Model 3: LASSO

model_2 <- cv.glmnet(x, y, alpha = 1)
best_lambda_2 <- model_2$lambda.min
model_LASSO <- glmnet(x, y, alpha = 1, lambda = best_lambda_2)
c_LASSO <- coef(model_LASSO)
y_LASSO <- c_LASSO[1] + c_LASSO[2]*data_test$Income + c_LASSO[3]*data_test$Limit + c_LASSO[4]*data_test$Rating + c_LASSO[5]*data_test$Cards + c_LASSO[6]*data_test$Age + c_LASSO[7]*data_test$Student
#View(non_negative_model(y_LASSO))



# Model 4: Elastic Net

model_3 <- cv.glmnet(x, y, alpha = .5)
best_lambda_3 <- model_3$lambda.min
model_EN <- glmnet(x, y, alpha = .5, lambda = best_lambda_3)
c_EN <- coef(model_EN)
y_EN <- c_EN[1] + c_EN[2]*data_test$Income + c_EN[3]*data_test$Limit + c_EN[4]*data_test$Rating + c_EN[5]*data_test$Cards + c_EN[6]*data_test$Age + c_EN[7]*data_test$Student 
#View(non_negative_model(y_EN))



# Model 4: Bayesian

model_bayesian <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student)
c_b <- coefficients(model_bayesian) 
y_b <- c_b[1] + c_b[2]*data_test$Income + c_b[3]*data_test$Limit + c_b[4]*data_test$Rating + c_b[5]*data_test$Cards + c_b[6]*data_test$Age + c_b[7]*data_test$Student
#View(non_negative_model(y_b))



# Model 5: Polynomial

model_poly <- lm(data = data_train, Balance ~ Income+Rating+Limit+Cards+Age+Student+I(Income^2)+I(Rating^2)+Student:Limit+Student:Rating+Student:Cards+Student:Age+Income:Student+Limit:Rating:Age+Limit:Rating:Student+Income:Rating:Limit+Income:Student:Age+Income:Rating:Limit:Student:Cards:Age) 
c_poly <- model_poly$coefficients
y_poly <- c_poly[1] + c_poly[2] * data_test$Income + c_poly[3] * data_test$Rating + c_poly[4] * data_test$Limit + c_poly[5] * data_test$Cards + c_poly[6] * data_test$Age + c_poly[7] * data_test$Student + c_poly[8] * (data_test$Income)^2 + c_poly[9] * (data_test$Rating)^2 +  c_poly[10] * data_test$Limit * data_test$Student + c_poly[11] * data_test$Rating * data_test$Student +  c_poly[12] * data_test$Cards * data_test$Student + c_poly[13] * data_test$Age * data_test$Student +  c_poly[14] * data_test$Income * data_test$Student +  c_poly[15] * data_test$Rating * data_test$Age * data_test$Limit + c_poly[16] * data_test$Student * data_test$Rating * data_test$Limit + c_poly[17] * data_test$Income * data_test$Rating * data_test$Limit +  c_poly[18] * data_test$Income * data_test$Age * data_test$Student + c_poly[19] * data_test$Income * data_test$Rating * data_test$Limit * data_test$Student * data_test$Age * data_test$Cards
#View(non_negative_model(y_poly))



# Model 6: Model Inform Model

y_mlr_pred <- predict(model_mlr, newx = x)
y_ridge_pred <- predict(model_ridge, newx = x)
y_LASSO_pred <- predict(model_LASSO, newx = x)
y_EN_pred <- predict(model_EN, newx = x)
y_B_pred <- predict(model_bayesian, newx = x)
y_Poly_pred <- predict(model_poly, newx = x)


data <- data_train %>%
  mutate(MLR = y_mlr_pred, Ridge = y_ridge_pred, LASSO = y_LASSO_pred, EN = y_EN_pred, B = y_B_pred, Poly = y_Poly_pred)


model_inform <- lm(data = data, Balance ~ Ridge + LASSO) 
c_inform <- model_inform$coefficients
y_inform <- c_inform[1] + c_inform[2] * y_ridge + c_inform[3] * y_LASSO
#View(non_negative_model(y_inform))

model_inform_2 <- lm(data = data, Balance ~ EN + B) 
c_inform_2 <- model_inform_2$coefficients
y_inform_2 <- c_inform_2[1] + c_inform_2[2] * y_EN + c_inform_2[3] * y_b
#View(non_negative_model(y_inform_2))



# Model 7: Model Polynomial Inform Model

model_poly_inform <- lm(data = data, Balance ~ MLR + EN + B + I(MLR^2) + I(B^2) + MLR:EN + MLR:B + EN:B + MLR:EN:B) 
c_poly_inform <- model_poly_inform$coefficients

y_poly_inform <- c_poly_inform[1] + c_poly_inform[2] * y_mlr + c_poly_inform[3] * y_EN + c_poly_inform[4] * y_b + c_poly_inform[5] * (y_mlr)^2 + c_poly_inform[6] * (y_b)^2 + c_poly_inform[7] * y_mlr * y_EN +  c_poly_inform[8] * y_mlr * y_b + c_poly_inform[9] * y_EN * y_b +  c_poly_inform[10] * y_mlr * y_EN * y_b
#View(non_negative_model(y_poly_inform))


# Model 8: Ridge Model Inform Model

x_2 <- as.matrix(data[c('MLR','EN','B','Poly')])


y_Poly_inform_pred <- predict(model_poly_inform, newx = x)

data <- data_train %>%
  mutate(MLR = y_mlr_pred, Ridge = y_ridge_pred, LASSO = y_LASSO_pred, EN = y_EN_pred, B = y_B_pred, Poly = y_Poly_pred,Poly_Inform = y_Poly_inform_pred)

x_3 <- as.matrix(data[c('MLR','EN','B','Poly','Poly_Inform')])

model_4 <- cv.glmnet(x_3,y, alpha = 0)
best_lambda_4 <- model_4$lambda.min
model_ridge_inform <- glmnet(x_3, y, alpha = 0, lambda = best_lambda_4)
c_ridge_inform <- coef(model_ridge_inform)

y_ridge_inform <- c_ridge_inform[1] + c_ridge_inform[2] * y_mlr + c_ridge_inform[3] * y_EN + c_ridge_inform[4] * y_b + c_ridge_inform[5] * y_poly + c_ridge_inform[6] * y_poly_inform 
#View(non_negative_model(y_ridge_inform))



# Model 9: LASSO Model Inform Model

model_5 <- cv.glmnet(x_2,y, alpha = 1)
best_lambda_5 <- model_5$lambda.min
model_LASSO_inform <- glmnet(x_2, y, alpha = 1, lambda = best_lambda_5)
c_LASSO_inform <- coef(model_LASSO_inform)

y_LASSO_inform <- c_LASSO_inform[1] + c_LASSO_inform[2] * y_mlr + c_LASSO_inform[3] * y_EN + c_LASSO_inform[4] * y_b + c_LASSO_inform[5] * y_poly
#View(non_negative_model(y_LASSO_inform))



# Model 10: EN Model Inform Model

model_6 <- cv.glmnet(x_2,y, alpha = .5)
best_lambda_6 <- model_6$lambda.min
model_EN_inform <- glmnet(x_2, y, alpha = .5, lambda = best_lambda_6)
c_EN_inform <- coef(model_EN_inform)

y_EN_inform <- c_EN_inform[1] + c_EN_inform[2] * y_mlr + c_EN_inform[3] * y_EN + c_EN_inform[4] * y_b + c_EN_inform[5] * y_poly
#View(non_negative_model(y_EN_inform))



# Model 11: LASSO Model Inform Model 2

x_3 <- as.matrix(data[c('MLR','EN','B','Poly','Poly_Inform')])

model_7 <- cv.glmnet(x_3,y, alpha = 1)
best_lambda_7 <- model_7$lambda.min
model_LASSO_inform_2 <- glmnet(x_3, y, alpha = 1, lambda = best_lambda_7)
c_LASSO_inform_2 <- coef(model_LASSO_inform_2)

y_LASSO_inform_2 <- c_LASSO_inform_2[1] + c_LASSO_inform_2[2] * y_mlr + c_LASSO_inform_2[3] * y_EN + c_LASSO_inform_2[4] * y_b + c_LASSO_inform_2[5] * y_poly + c_LASSO_inform_2[6] * y_poly_inform
#View(non_negative_model(y_LASSO_inform_2))



# Model 12: EN Model Inform Model 2

model_8 <- cv.glmnet(x_3,y, alpha = .5)
best_lambda_8 <- model_8$lambda.min
model_EN_inform_2 <- glmnet(x_3, y, alpha = .5, lambda = best_lambda_8)
c_EN_inform_2 <- coef(model_EN_inform_2)

y_EN_inform_2 <- c_EN_inform_2[1] + c_EN_inform_2[2] * y_mlr + c_EN_inform_2[3] * y_EN + c_EN_inform_2[4] * y_b + c_EN_inform_2[5] * y_poly + c_EN_inform_2[6] * y_poly_inform
#View(non_negative_model(y_EN_inform_2))



# Model 13: Ridge Model Inform Model 2

y_LASSO_inform_2_pred <- predict(model_LASSO_inform_2, newx = x_3)
y_EN_inform_2_pred <- predict(model_EN_inform_2, newx = x_3)

data <- data %>% mutate(LASSO_INFORM_2 = y_LASSO_inform_2_pred, EN_INFORM_2 =  y_EN_inform_2_pred)

x_4 <- as.matrix(data[c('Poly_Inform', 'LASSO_INFORM_2', 'EN_INFORM_2')])

model_9 <- cv.glmnet(x_4,y, alpha = 0)
best_lambda_9 <- model_9$lambda.min
model_ridge_inform_2 <- glmnet(x_4, y, alpha = 0, lambda = best_lambda_9)
c_ridge_inform_2 <- coef(model_ridge_inform_2)

y_ridge_inform_2 <- c_ridge_inform_2[1] + c_ridge_inform_2[2] * y_poly_inform + c_ridge_inform_2[3] * y_LASSO_inform_2 + c_ridge_inform_2[4] * y_EN_inform_2 
#View(non_negative_model(y_ridge_inform_2))




# Model 14: Combination Inform Model 2

y_Ridge_inform_2_pred <- predict(model_ridge_inform_2, newx = x_4)
data <- data %>% mutate(Ridge_INFORM_2 = y_Ridge_inform_2_pred)

model_poly_inform_2 <- lm(data = data, Balance ~ Poly_Inform + EN_INFORM_2 + I(LASSO_INFORM_2^2) + I(EN_INFORM_2^2) + I(Ridge_INFORM_2^2) + Poly_Inform:EN_INFORM_2:Ridge_INFORM_2 + Poly_Inform:EN_INFORM_2:LASSO_INFORM_2 + Poly_Inform:LASSO_INFORM_2:Ridge_INFORM_2) 
c_poly_inform_2 <- model_poly_inform_2$coefficients

y_poly_inform_2 <- c_poly_inform_2[1] + c_poly_inform_2[2] * y_poly_inform + c_poly_inform_2[3] * y_EN_inform_2 + c_poly_inform_2[4] * (y_LASSO_inform_2)^2 + c_poly_inform_2[5] * (y_EN_inform_2)^2 + c_poly_inform_2[6] * (y_ridge_inform_2)^2 + c_poly_inform_2[7] * y_poly_inform * y_EN_inform_2 * y_ridge_inform_2 +  c_poly_inform_2[8] * y_poly_inform * y_EN_inform_2 * y_LASSO_inform_2 + c_poly_inform_2[9] * y_poly_inform *  y_ridge_inform_2 * y_LASSO_inform_2
#View(non_negative_model(y_poly_inform_2))



# Tried to add the last interaction between the four models still very good but the above model is slightly better
# model_poly_inform_2 <- lm(data = data, Balance ~ Poly_Inform + EN_INFORM_2 + I(LASSO_INFORM_2^2) + I(EN_INFORM_2^2) + I(Ridge_INFORM_2^2) + Poly_Inform:EN_INFORM_2:Ridge_INFORM_2 + Poly_Inform:EN_INFORM_2:LASSO_INFORM_2 + Poly_Inform:LASSO_INFORM_2:Ridge_INFORM_2 + LASSO_INFORM_2:EN_INFORM_2:Ridge_INFORM_2:Poly_Inform) 
# c_poly_inform_2 <- model_poly_inform_2$coefficients
# 
# y_poly_inform_2 <- c_poly_inform_2[1] + c_poly_inform_2[2] * y_poly_inform + c_poly_inform_2[3] * y_EN_inform_2 + c_poly_inform_2[4] * (y_LASSO_inform_2)^2 + c_poly_inform_2[5] * (y_EN_inform_2)^2 + c_poly_inform_2[6] * (y_ridge_inform_2)^2 + c_poly_inform_2[7] * y_poly_inform * y_EN_inform_2 * y_ridge_inform_2 +  c_poly_inform_2[8] * y_poly_inform * y_EN_inform_2 * y_LASSO_inform_2 + c_poly_inform_2[9] * y_poly_inform *  y_ridge_inform_2 * y_LASSO_inform_2 + c_poly_inform_2[10] * y_poly_inform *  y_ridge_inform_2 * y_LASSO_inform_2 * y_EN_inform_2
# #View(non_negative_model(y_poly_inform_2))



# Checking the adequacy of Polynomial Inform 2: Assumptions Satisfied

par(mfrow = c(2,2))
plot(model_poly_inform_2)


# Checking Normality Assumption: 
y_poly_inform_2_resid <- resid(model_poly_inform_2, newx = x_4)
y_poly_inform_2_pred <- predict(model_poly_inform_2, newx = x_4)

shapiro.test(y_poly_inform_2_resid)$p.value   # Cannot reject the null hypothesis (Not normal)


ggplot(data = data) + 
  geom_point(aes(x = Income, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = y_poly_inform_2_pred), se = F) +
  facet_wrap(~Student) +
  labs(color = 'Number of Cards')

ggplot(data = data) + 
  geom_point(aes(x = Limit, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = y_poly_inform_2_pred), se = F) +
  facet_wrap(~Student) +
  labs(color = 'Number of Cards')

ggplot(data = data) + 
  geom_point(aes(x = Rating, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = y_poly_inform_2_pred), se = F) +
  facet_wrap(~Student) +
  labs(color = 'Number of Cards')




# Checking the adequacy of Ridge Inform 2: There is a non linear relationship between y_hat & resid

par(mfrow = c(1,1))

y_ridge_inform_2_resid <- data$Balance - y_ridge_inform_2_pred
qqnorm(y_ridge_inform_2_resid)
qqline(y_ridge_inform_2_resid)


# Checking Normality Assumption: 

shapiro.test(y_ridge_inform_2_resid)$p.value   # Cannot reject the null hypothesis


ggplot(data = data, aes(x = y_ridge_inform_2_resid)) +
  geom_histogram(bins = 100) +
  xlab('Residual') + 
  ylab('Frequency') +
  ggtitle('Normality of Model : Ridge Inform 2')

ggplot(data = data, aes(x = y_ridge_inform_2_pred, y = y_ridge_inform_2_resid)) +
  geom_point() +
  geom_ref_line(h= 0, size = 1, colour = 'white') +
  geom_smooth(se = F, color = 'orange') +
  xlab('Prediction') + 
  ylab('Residual') +
  ggtitle('Model : Ridge Inform 2')



# Checking the adequacy of LASSO Inform 2 : There is a non linear relationship between y_hat & resid

y_LASSO_inform_2_resid <- data$Balance - y_LASSO_inform_2_pred
qqnorm(y_LASSO_inform_2_resid)
qqline(y_LASSO_inform_2_resid)


# Checking Normality Assumption: 

shapiro.test(y_LASSO_inform_2_resid)$p.value   # Cannot reject the null hypothesis

ggplot(data = data, aes(x = y_LASSO_inform_2_resid)) +
  geom_histogram(bins = 100) +
  xlab('Residual') + 
  ylab('Frequency') +
  ggtitle('Normality of Model : LASSO Inform 2')

ggplot(data = data, aes(x = y_LASSO_inform_2_pred, y = y_LASSO_inform_2_resid)) +
  geom_point() +
  geom_ref_line(h= 0, size = 1, colour = 'white') +
  geom_smooth(se = F, color = 'orange') +
  xlab('Prediction') + 
  ylab('Residual') +
  ggtitle('Model : LASSO Inform 2')



# Checking the adequacy of Ridge Inform 2: There is a non linear relationship between y_hat & resid

y_EN_inform_2_resid <- data$Balance - y_EN_inform_2_pred
qqnorm(y_EN_inform_2_resid)
qqline(y_EN_inform_2_resid)


# Checking Normality Assumption: 

shapiro.test(y_EN_inform_2_resid)$p.value   # Cannot reject the null hypothesis


ggplot(data = data, aes(x = y_EN_inform_2_resid)) +
  geom_histogram(bins = 100) +
  xlab('Residual') + 
  ylab('Frequency') +
  ggtitle('Normality of Model : EN Inform 2')

ggplot(data = data, aes(x = y_EN_inform_2_pred, y = y_EN_inform_2_resid)) +
  geom_point() +
  geom_ref_line(h= 0, size = 1, colour = 'white') +
  geom_smooth(se = F, color = 'orange') +
  xlab('Prediction') + 
  ylab('Residual') +
  ggtitle('Model : EN Inform 2')
