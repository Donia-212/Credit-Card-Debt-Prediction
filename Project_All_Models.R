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

data_test$Own <- ifelse(data_test$Own == "Yes",1,0)
data_test$Student <- ifelse(data_test$Student == "Yes",1,0)
data_test$Married <- ifelse(data_test$Married == "Yes",1,0)

data_test <- data_test %>%
  mutate(Region = case_when(
    Region == "South" ~ 1,
    Region == "East"  ~ 2,
    Region == "West"  ~ 3
  ))

data_test <- data_test[-1]


attach(data_test)
Student <- factor(data_test$Student)
Own <- factor(data_test$Own)
Region <- factor(data_test$Region)
Married <- factor(data_test$Married)


attach(data_train)
Student <- factor(data_train$Student)
Own <- factor(data_train$Own)
Region <- factor(data_train$Region)
Married <- factor(data_train$Married)



non_negative_model <- function(m){
  for(i in seq(1,length(m))){
    if(m[i]< 0){
      m[i] = 0
    }
  }
  return(as_tibble(m))
}


# ----------------------------------------------------------------------------------------------------------------------------------


# Model 1: Mars Model: 

x_train <- as.matrix(data_train[c('Income','Limit','Rating','Cards','Age','Student')])
x <- cbind(x_train, Income_Limit = c(data_train$Income*data_train$Limit), Income_Rating = c(data_train$Income*data_train$Rating), Rating_Limit = c(data_train$Rating*data_train$Limit), Age_Limit = c(data_train$Age*data_train$Limit), Student_Limit = c(data_train$Student*data_train$Limit))

data_test_modified <- data_test %>% mutate(Income_Limit = data_test$Income*data_test$Limit, Income_Rating = data_test$Income*data_test$Rating, Rating_Limit = data_test$Rating*data_test$Limit, Age_Limit = data_test$Age*data_test$Limit, Student_Limit = data_test$Student*data_test$Limit) %>% dplyr::select(-Own,-Married,-Region,-Education) 
y <- data_train$Balance
x_test <- as.matrix(data_test_modified[,])

parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))

set.seed(8885)
cv_mars_model <- train(x = x, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars <- predict(object = cv_mars_model$finalModel, newdata = data_test_modified) 
# View(non_negative_model(y_mars))
# summary(cv_mars_model$finalModel)
# shapiro.test(resid(cv_mars_model$finalModel))
# par(mfrow = c(2,2)); plot(cv_mars_model$finalModel)



# ----------------------------------------------------------------------------------------------------------------------------------


# Model 2: EN Model

x_2 <- cbind(x_train,  Income_Rating = c(data_train$Income*data_train$Rating), Rating_Limit = c(data_train$Rating*data_train$Limit), Rating_Student = c(data_train$Rating*data_train$Student), Rating_Limit_Student = c(data_train$Rating*data_train$Limit*data_train$Student), Rating_Limit_Income_Student = c(data_train$Rating*data_train$Income*data_train$Limit*data_train$Student))

data_test_modified_2 <- data_test %>% dplyr::select(-Own,-Married,-Region,-Education) %>% mutate( Income_Rating = data_test$Income*data_test$Rating, Rating_Limit = data_test$Rating*data_test$Limit, Rating_Student = data_test$Rating*data_test$Student, Rating_Limit_Student = data_test$Rating*data_test$Limit*data_test$Student, Rating_Limit_Income_Student = data_test$Rating*data_test$Income*data_test$Limit*data_test$Student)
x_test_2 <- as.matrix(data_test_modified_2[,])

parameter_grid_en <- expand.grid(alpha = seq(0,1, length = 5), lambda = seq(0.01,0.9, length = 10))
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 5, search = "random", verboseIter = TRUE)
cv_en<- train(x_2,y, method = "glmnet", metric = 'Rsquared', trControl = train_control, tuneGrid = parameter_grid_en, seed = 8885)
model_EN <- glmnet(x_2, y, alpha = cv_en$bestTune[1], lambda = cv_en$bestTune[2])
y_EN <- predict(model_EN, newx = x_test_2)
# View(non_negative_model(y_EN))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 3: MLR Model

data_test_modified <- data_test %>% dplyr::select(1:5, Student)
model_mlr <- lm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student + Income:Limit + Income:Rating + Rating:Limit + Age:Limit + Student:Limit + Age:Student + Student:Income + Student:Rating + Student:Rating:Limit + Student:Rating:Limit:Income)
y_mlr <- predict(model_mlr, data_test)
# View(non_negative_model(y_mlr))
# shapiro.test(rstandard(model_mlr))
# par(mfrow = c(2,2)); plot(model_mlr)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 4: LASSO Model

model_LASSO <- cv.glmnet(x_2, y, alpha = 1, seed = 8885)
best_lambda_LASSO <- model_LASSO$lambda.min
model_LASSO <- glmnet(x_2, y, alpha = 1, lambda = best_lambda_LASSO)

y_LASSO <- predict(model_LASSO, newx = x_test_2)
# View(non_negative_model(y_LASSO))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 5: Polynomial

model_poly <- lm(data = data_train, Balance ~ poly(Income, 2, raw = T) + poly(Rating,2, raw = T) + poly(Cards, 2, raw = T) + poly(Student, 3,raw = T) + Income:Limit + Income:Rating + Limit:Rating + Limit:Age + Limit:Student + Income:Student + Rating:Student + Limit:Rating:Student + Income:Limit:Rating:Student)
y_poly <- predict(model_poly, data_test)
# View(non_negative_model(y_poly))
# shapiro.test(rstandard(model_poly))
# par(mfrow = c(2,2)); plot(model_poly)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 6: Root Polynomial

model_root <- lm(data = data_train, Balance ~ log(Income) + sqrt(Rating) + I(Limit^(1/3)) + sqrt(Cards) + sqrt(Student) + sqrt(Income * Limit) + sqrt(Income * Rating) + I((Limit * Rating)^(1/3)) + sqrt(Limit * Age) + sqrt(Limit * Student) + sqrt(Income * Student) + I((Rating * Student)^(1/3)) + sqrt(Limit * Rating * Student) + sqrt(Income * Limit * Rating * Student))
y_root <- predict(model_root, data_test)
# View(non_negative_model(y_root))
# shapiro.test(rstandard(model_root))
# par(mfrow = c(2,2)); plot(model_root)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 7: LOESS

parameter_grid_loess <- expand.grid(span = seq(0.5, 0.9, len = 5), degree = 1)
cv_loess_model <- train(x_2, y, method = "gamLoess", metric = 'Rsquared', tuneGrid = parameter_grid_loess, trControl = trainControl(method = "cv"), seed = 8885)
y_loess <- predict(cv_loess_model, x_test_2)
# View(non_negative_model(y_loess))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 8: Robust

model_robust <- rlm(data = data_train, Balance ~ Income + Rating + Limit + Cards + Age + Student)
y_robust <- predict(model_robust,  data_test)
# View(non_negative_model(y_robust))
# shapiro.test(resid(model_robust))
# par(mfrow = c(2,2)); plot(model_robust)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9: Inform Linear

y_mars_pred <- predict(object = cv_mars_model$finalModel, newx = x)
y_EN_pred <- predict(model_EN, newx = x_2)
y_mlr_pred <- model_mlr$fitted.values
y_LASSO_pred <- predict(model_LASSO, newx = x_2)
y_poly_pred <- model_poly$fitted.values
y_root_pred <- model_root$fitted.values
y_LOESS_pred <- predict(object = cv_loess_model$finalModel, newx = x_2)
y_robust_pred <- model_robust$fitted.values


data <- data_train %>%
  mutate(MARS = y_mars_pred, EN = y_EN_pred, MLR = y_mlr_pred, LASSO = y_LASSO_pred, Poly = y_poly_pred, Root = y_root_pred, LOESS = y_LOESS_pred, Robust = y_robust_pred)

data_test_modified_3 <- data_test %>%
  mutate(MARS = y_mars, EN = y_EN, MLR = y_mlr, LASSO = y_LASSO, Poly = y_poly, Root = y_root, LOESS = y_loess, Robust = y_robust)


options(scipen = 999)
model_linear_inform <- lm(data = data, Balance ~ Root*Poly*MLR + MARS*LASSO*Poly*Robust + LOESS*EN*MARS*Root + LOESS*MLR*Root*EN + Root*MARS*MLR) 
y_linear_inform <- predict(model_linear_inform, newdata = data_test_modified_3)
# View(non_negative_model(y_linear_inform))
# shapiro.test(rstandard(model_linear_inform))
# par(mfrow = c(2,2)); plot(model_linear_inform)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9: Inform Polynomial

model_poly_inform <- lm(data = data, Balance ~ poly(MARS,2,raw = T) + poly(EN,2,raw = T) + poly(MLR,3,raw = T) + poly(LASSO,2,raw = T) + poly(LOESS,2,raw = T) + poly(Poly,2,raw = T) + poly(Root,3,raw = T) + poly(Robust,2,raw = T) + I(Root^2)*MARS*Poly*I(MLR^2)*LOESS*LASSO*EN*Robust) 
y_poly_inform <- predict(model_poly_inform, newdata = data_test_modified_3)
# View(non_negative_model(y_poly_inform))
# shapiro.test(rstandard(model_poly_inform))
# par(mfrow = c(2,2)); plot(model_poly_inform)


# model_poly_inform_2 <- lm(data = data, Balance ~ poly(MARS,2,raw = T) + poly(EN,2,raw = T) + poly(MLR,3,raw = T) + poly(LASSO,2,raw = T) + poly(LOESS,2,raw = T) + poly(Poly,2,raw = T) + poly(Root,3,raw = T) + poly(Robust,2,raw = T) + poly(MLR,2,raw = T)*MARS*Poly*poly(Root,2,raw = T)*LOESS*LASSO*EN*Robust) 
# y_poly_inform_2 <- predict(model_poly_inform_2, newdata = data_test_modified_3)
# View(non_negative_model(y_poly_inform_2))
# shapiro.test(rstandard(model_poly_inform_2))
# par(mfrow = c(2,2)); plot(model_poly_inform_2)


# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9: Inform Root


non_negative_model_2 <- function(m){
  for(i in seq(1,length(m))){
    if(m[i]< 0){
      m[i] = 0
    }
  }
  return(m)
}

y_mars_pred_2 <- non_negative_model_2(predict(object = cv_mars_model$finalModel, newx = x))
y_EN_pred_2 <- non_negative_model_2(predict(model_EN, newx = x_2))
y_mlr_pred_2 <- non_negative_model_2(model_mlr$fitted.values)
y_LASSO_pred_2 <- non_negative_model_2(predict(model_LASSO, newx = x_2))
y_poly_pred_2 <- non_negative_model_2(model_poly$fitted.values)
y_root_pred_2 <- non_negative_model_2(model_root$fitted.values)
y_LOESS_pred_2 <- non_negative_model_2(predict(object = cv_loess_model$finalModel, newx = x_2))
y_robust_pred_2 <- non_negative_model_2(model_robust$fitted.values)



data_2 <- data_train %>%
  mutate(MARS = y_mars_pred_2, EN = y_EN_pred_2, MLR = y_mlr_pred_2, LASSO = y_LASSO_pred_2, Poly = y_poly_pred_2, Root = y_root_pred_2, LOESS = y_LOESS_pred_2, Robust = y_robust_pred_2)

data_test_modified_4 <- data_test %>%
  mutate(MARS = non_negative_model_2(y_mars), EN = non_negative_model_2(y_EN), MLR = non_negative_model_2(y_mlr), LASSO = non_negative_model_2(y_LASSO), Poly = non_negative_model_2(y_poly), Root = non_negative_model_2(y_root), LOESS = non_negative_model_2(y_loess), Robust = non_negative_model_2(y_robust))



model_root_inform <- lm(data = data_2, sqrt(Balance) ~ sqrt(MARS) + sqrt(EN) + sqrt(MLR) + sqrt(LASSO) + sqrt(LOESS) + sqrt(Poly) + sqrt(Root) + sqrt(Robust) + poly(Root,2,raw = T)*MARS*Poly*poly(MLR,2,raw = T)*LOESS*LASSO*EN*Robust)
summary(model_root_inform)
y_root_inform <- (predict(model_root_inform, newdata = data_test_modified_4))^(2)
# View(as_tibble(y_root_inform))
# shapiro.test(rstandard(model_root_inform))    Although it is not normal but it fixed the funnel shape of the residuals
# par(mfrow = c(2,2)); plot(model_root_inform)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9: Inform Linear

model_linear_transform <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_linear_transform <- (predict(model_linear_transform, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform))
# shapiro.test(rstandard(model_linear_transform))
# par(mfrow = c(2,2)); plot(model_linear_transform)

data_3 <- data_2 %>% mutate(Balance = Balance + 1.5)
model_boxcox <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_boxcox <- (predict(model_boxcox, newdata = data_test_modified_4))^2
# summary(model_boxcox)  

bc <- boxcox(model_boxcox)
lambda <- bc$x[which.max(bc$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox <- lm(data = data_3, ((sqrt(Balance)^lambda - 1)/lambda) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_linear_transformed_boxcox <- (lambda * predict(model_linear_transformed_boxcox, newdata = data_test_modified_4) + 1) ^ (2/ lambda) 
y_linear_transformed_boxcox <- replace(y_linear_transformed_boxcox, is.nan(y_linear_transformed_boxcox), 0)
# View(as_tibble(y_linear_transformed_boxcox))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox)




# All the varaiable sqrt individually (Same results)
# data_3 <- data_2 %>% mutate(Balance = Balance + 1.5)
# model_boxcox <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root)*sqrt(Poly)*sqrt(MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
# y_boxcox <- (predict(model_boxcox, newdata = data_test_modified_4))^2
# summary(model_boxcox)  
# bc <- boxcox(model_boxcox)
# lambda <- bc$x[which.max(bc$y)]
# model_linear_transformed_boxcox <- lm(data = data_3, ((sqrt(Balance)^lambda - 1)/lambda) ~ sqrt(Root)*sqrt(Poly)*sqrt(MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
# y_linear_transformed_boxcox <- (lambda * predict(model_linear_transformed_boxcox, newdata = data_test_modified_4) + 1) ^ (2/ lambda) 
# y_linear_transformed_boxcox <- replace(y_linear_transformed_boxcox, is.nan(y_linear_transformed_boxcox), 0)
# View(as_tibble(y_linear_transformed_boxcox))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox)


# We will try it again with the optimal choice of Balance (Same results)
# data_3 <- data_2 %>% mutate(Balance = Balance + 1.38)
# model_boxcox <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
# y_boxcox <- (predict(model_boxcox, newdata = data_test_modified_4))^2
# summary(model_boxcox)  
# bc <- boxcox(model_boxcox)
# lambda <- bc$x[which.max(bc$y)]
# model_linear_transformed_boxcox <- lm(data = data_3, ((sqrt(Balance)^lambda - 1)/lambda) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
# y_linear_transformed_boxcox <- (lambda * predict(model_linear_transformed_boxcox, newdata = data_test_modified_4) + 1) ^ (2/ lambda) 
# y_linear_transformed_boxcox <- replace(y_linear_transformed_boxcox, is.nan(y_linear_transformed_boxcox), 0)
# View(as_tibble(y_linear_transformed_boxcox))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox)



model_linear_transform_2 <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root*LOESS)*sqrt(MLR*MARS)*sqrt(LASSO*Robust)*sqrt(Poly*EN)) 
y_linear_transform_2 <- (predict(model_linear_transform_2, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_2))
# shapiro.test(rstandard(model_linear_transform_2))
# par(mfrow = c(2,2)); plot(model_linear_transform_2)


model_boxcox_2 <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*LOESS)*sqrt(MLR*MARS)*sqrt(LASSO*Robust)*sqrt(Poly*EN)) 
y_boxcox_2 <- (predict(model_boxcox_2, newdata = data_test_modified_4))^2
# summary(model_boxcox_2)  

bc_3 <- boxcox(model_boxcox_2)
lambda_3 <- bc_3$x[which.max(bc_3$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_2 <- lm(data = data_3, ((sqrt(Balance)^lambda_3 - 1)/lambda_3) ~ sqrt(Root*LOESS)*sqrt(MLR*MARS)*sqrt(LASSO*Robust)*sqrt(Poly*EN)) 
y_linear_transformed_boxcox_2 <- (lambda_3 * predict(model_linear_transformed_boxcox_2, newdata = data_test_modified_4) + 1) ^ (2/ lambda_3) 
y_linear_transformed_boxcox_2 <- replace(y_linear_transformed_boxcox_2, is.nan(y_linear_transformed_boxcox_2), 0)
# View(as_tibble(y_linear_transformed_boxcox_2))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_2)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 10: Inform Combined

y_Linear_pred <- model_linear_inform$fitted.values
y_Poly_pred <- model_poly_inform$fitted.values
y_Root_pred <- (model_root_inform$fitted.values)^2
y_Linear_Transformed_pred <- (lambda * model_linear_transformed_boxcox$fitted.values + 1) ^ (2/lambda)
y_Linear_Transformed_pred <- replace(y_Linear_Transformed_pred, is.nan(y_Linear_Transformed_pred), 0)

data_4 <- data_2 %>% mutate(Linear = y_Linear_pred, Poly_Inform = y_Poly_pred, Root_Inform = y_Root_pred, Linear_Transformed = y_Linear_Transformed_pred) %>% dplyr::select(-Balance)
data_test_modified_5 <- data_test_modified_4 %>% mutate(Linear = y_linear_inform, Poly_Inform = y_poly_inform, Root_Inform = y_root_inform, Linear_Transformed = y_linear_transformed_boxcox)

x_3 <- as.matrix(data_4[,])
x_test_3 <- as.matrix(data_test_modified_5[,])

parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))
set.seed(8885)
cv_mars_model_2 <- train(x = x_3, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars_2 <- predict(object = cv_mars_model_2$finalModel, newdata = data_test_modified_5) 
# View(non_negative_model(y_mars_2))
# par(mfrow = c(2,2)); plot(cv_mars_model_2$finalModel)


y_MARS_2_pred <- predict(object = cv_mars_model_2$finalModel, newx = x_3)
data_4 <- data_4 %>% mutate(MARS_2 = y_MARS_2_pred)
data_test_modified_5 <- data_test_modified_5 %>% mutate(MARS_2 = y_mars_2)


model_combined <- lm(data = data_4, Balance ~ Poly_Inform + MARS_2) 
y_combined <- predict(model_combined, newdata = data_test_modified_5)
# View(non_negative_model(y_combined))
# shapiro.test(rstandard(model_combined))
# par(mfrow = c(2,2)); plot(model_combined)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 11: Inform Bayesian

model_b <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student + Income:Limit + Income:Rating + Rating:Limit + Age:Limit + Student:Limit + Age:Student + Student:Income + Student:Rating + Student:Rating:Limit + Student:Rating:Limit:Income, seed = 8885)
y_b <- predict(model_b, newdata = data_test)
# View(non_negative_model(y_b))
# shapiro.test(resid(model_b))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9: Inform Linear with Bayesian

y_B_pred <- model_b$fitted.values
data_5 <- data %>% mutate(B = y_B_pred)
data_test_modified_6 <- data_test_modified_3 %>% mutate(B = y_b)

model_linear_inform_b <- lm(data = data_5, Balance ~ B*Root*Poly*MLR + MARS*LASSO*Poly*Robust*B + B*LOESS*EN*MARS*Root + LOESS*MLR*Root*EN + Root*MARS*MLR) 
y_linear_inform_b <- predict(model_linear_inform_b, newdata = data_test_modified_6)
# View(non_negative_model(y_linear_inform_b))
# shapiro.test(rstandard(model_linear_inform_b))
# par(mfrow = c(2,2)); plot(model_linear_inform_b)

# BoxCox to make variance more constant
y_b_pred_2 <- non_negative_model_2(model_b$fitted.values)
data_6 <- data_2 %>% mutate(B = y_b_pred_2)
data_test_modified_7 <- data_test_modified_4 %>% mutate(B = non_negative_model_2(y_b))

model_linear_transform_b <- lm(data = data_6, sqrt(Balance) ~ sqrt(B*Root*Poly*MLR) + sqrt(MARS*LASSO*Poly*Robust*B) + sqrt(B*LOESS*EN*MARS*Root) + sqrt(LOESS*MLR*Root*EN) + sqrt(Root*MARS*MLR)) 
y_linear_transform <- (predict(model_linear_transform, newdata = data_test_modified_7))^2
# View(non_negative_model(y_linear_transform))
# shapiro.test(rstandard(model_linear_transform))
# par(mfrow = c(2,2)); plot(model_linear_transform)

data_7 <- data_6 %>% mutate(Balance = Balance + 1.5)
model_boxcox_b <- lm(data = data_7, sqrt(Balance) ~ sqrt(B)*sqrt(Root)*sqrt(Poly)*sqrt(MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B) + sqrt(B)*sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_boxcox_b <- (predict(model_boxcox_b, newdata = data_test_modified_7))^2

bc_2 <- boxcox(model_boxcox_b)
lambda_2 <- bc_2$x[which.max(bc_2$y)]

model_linear_transformed_boxcox_b <- lm(data = data_7, ((sqrt(Balance)^lambda_2 - 1)/lambda_2) ~ sqrt(B)*sqrt(Root)*sqrt(Poly)*sqrt(MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B) + sqrt(B)*sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR))
y_linear_transformed_boxcox_b <- (lambda_2 * predict(model_linear_transformed_boxcox_b, newdata = data_test_modified_7) + 1) ^ (2/ lambda_2) 
y_linear_transformed_boxcox_b <- replace(y_linear_transformed_boxcox_b, is.nan(y_linear_transformed_boxcox_b), 0)
# View(as_tibble(y_linear_transformed_boxcox_b))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_b)