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

data_train$Student <- factor(data_train$Student)
data_train$Own <- factor(data_train$Own)
data_train$Region <- factor(data_train$Region)
data_train$Married <- factor(data_train$Married)




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


data_test$Student <- factor(data_test$Student)
data_test$Own <- factor(data_test$Own)
data_test$Region <- factor(data_test$Region)
data_test$Married <- factor(data_test$Married)


data_train <- data_train %>% mutate(Balance = sqrt(Balance), Rating = sqrt(Rating), Limit = sqrt(Limit))
data_test <- data_test %>% mutate(Rating = sqrt(Rating), Limit = sqrt(Limit))
options(scipen = 999)



# ----------------------------------------------------------------------------------------------------------------------------------


# Model 1: Mars Model: 

x_train <- as.matrix(data_train[c('Income','Limit','Rating','Cards','Age','Student')])
x <- as.matrix(data_train[,])
x <- cbind(x_train, Income_Limit = c(data_train$Income*data_train$Limit), Income_Rating = c(data_train$Income*data_train$Rating), Rating_Limit = c(data_train$Rating*data_train$Limit), Age_Limit = c(data_train$Age*data_train$Limit), Student_Limit = c(data_train$Student*data_train$Limit))

data_test_modified <- data_test %>% mutate(Income_Limit = data_test$Income*data_test$Limit, Income_Rating = data_test$Income*data_test$Rating, Rating_Limit = data_test$Rating*data_test$Limit, Age_Limit = data_test$Age*data_test$Limit, Student_Limit = data_test$Student*data_test$Limit) %>% dplyr::select(-Own,-Married,-Region,-Education) 
y <- data_train$Balance
x_test <- as.matrix(data_test_modified[,])

parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))

set.seed(8885)
cv_mars_model <- train(x = x, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars <- (predict(object = cv_mars_model$finalModel, newdata = data_test_modified))^2 
# summary(cv_mars_model$finalModel)
# View(as_tibble(y_mars))



# ----------------------------------------------------------------------------------------------------------------------------------


# Model 2: EN Model

x_2 <- cbind(x_train,  Income_Rating = c(data_train$Income*data_train$Rating), Rating_Limit = c(data_train$Rating*data_train$Limit), Rating_Student = c(data_train$Rating*data_train$Student), Rating_Limit_Student = c(data_train$Rating*data_train$Limit*data_train$Student), Rating_Limit_Income_Student = c(data_train$Rating*data_train$Income*data_train$Limit*data_train$Student))

data_test_modified_2 <- data_test %>% dplyr::select(-Own,-Married,-Region,-Education) %>% mutate( Income_Rating = data_test$Income*data_test$Rating, Rating_Limit = data_test$Rating*data_test$Limit, Rating_Student = data_test$Rating*data_test$Student, Rating_Limit_Student = data_test$Rating*data_test$Limit*data_test$Student, Rating_Limit_Income_Student = data_test$Rating*data_test$Income*data_test$Limit*data_test$Student)
x_test_2 <- as.matrix(data_test_modified_2[,])

parameter_grid_en <- expand.grid(alpha = seq(0,1, length = 5), lambda = seq(0.01,0.9, length = 10))

train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              verboseIter = TRUE)

cv_en<- train(x_2,y, method = "glmnet", metric = 'Rsquared', trControl = train_control, tuneGrid = parameter_grid_en, seed = 8885)
model_EN <- glmnet(x_2, y, alpha = cv_en$bestTune[1], lambda = cv_en$bestTune[2])
y_EN <- (predict(model_EN, newx = x_test_2))^2
# View(as_tibble(y_EN))




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 3: MLR Model

data_test_modified <- data_test %>% dplyr::select(1:5, Student)
model_mlr <- lm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student + Income*Rating*Limit*Student*Cards*Age)
y_mlr <- (predict(model_mlr, data_test))^2
# View(non_negative_model(y_mlr))
# summary(model_mlr)

stepAIC(model_mlr, direction = 'both')



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 4: LASSO Model

model_LASSO <- cv.glmnet(x_2, y, alpha = 1, seed = 8885)
best_lambda_LASSO <- model_LASSO$lambda.min
model_LASSO <- glmnet(x_2, y, alpha = 1, lambda = best_lambda_LASSO)

y_LASSO <- predict(model_LASSO, newx = x_test_2)
# View(non_negative_model(y_LASSO))
# sum(y_LASSO < 0)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 5: Polynomial

model_poly <- lm(data = data_train, Balance ~ poly(Income, 2, raw = T) + poly(Rating,2, raw = T) + poly(Cards, 2, raw = T) + poly(Student, 3,raw = T) + Income:Limit + Income:Rating + Limit:Rating + Limit:Age + Limit:Student + Income:Student + Rating:Student + Limit:Rating:Student + Income:Limit:Rating:Student)

y_poly <- predict(model_poly, data_test)
# View(non_negative_model(y_poly))
# summary(model_poly)
# sum(y_poly < 0)




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 6: Root Polynomial

model_root <- lm(data = data_train, Balance ~ log(Income) + sqrt(Rating) + I(Limit^(1/3)) + sqrt(Cards) + sqrt(Student) + sqrt(Income * Limit) + sqrt(Income * Rating) + I((Limit * Rating)^(1/3)) + sqrt(Limit * Age) + sqrt(Limit * Student) + sqrt(Income * Student) + I((Rating * Student)^(1/3)) + sqrt(Limit * Rating * Student) + sqrt(Income * Limit * Rating * Student))

y_root <- predict(model_root, data_test)
# View(non_negative_model(y_root))
# summary(model_root)
# sum(y_root < 0)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 7: LOESS

parameter_grid_loess <- expand.grid(span = seq(0.5, 0.9, len = 5), degree = 1)
cv_loess_model <- train(x_2, y, method = "gamLoess", metric = 'Rsquared', tuneGrid = parameter_grid_loess, trControl = trainControl(method = "cv"), seed = 8885)
y_loess <- predict(cv_loess_model, x_test_2)
# View(non_negative_model(y_loess))
# sum(y_loess < 0)




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 7: Bayesian

# model_b <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student + Income:Limit + Income:Rating + Rating:Limit + Age:Limit + Student:Limit + Age:Student + Student:Income + Student:Rating + Student:Rating:Limit + Student:Rating:Limit:Income, seed = 8885)
# y_b <- predict(model_b, newdata = data_test)
# # View(non_negative_model(y_b))
# # sum(y_b < 0)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 8: Inform

y_mars_pred <- predict(object = cv_mars_model$finalModel, newx = x)
y_EN_pred <- predict(model_EN, newx = x_2)
y_mlr_pred <- model_mlr$fitted.values
y_LASSO_pred <- predict(model_LASSO, newx = x_2)
y_poly_pred <- model_poly$fitted.values
y_root_pred <- model_root$fitted.values
y_LOESS_pred <- predict(object = cv_loess_model$finalModel, newx = x_2)
# y_B_pred <- model_b$fitted.values


# data <- data_train %>%
#   mutate(MARS = y_mars_pred, EN = y_EN_pred, MLR = y_mlr_pred, LASSO = y_LASSO_pred, Poly = y_poly_pred, Root = y_root_pred, LOESS = y_LOESS_pred, B = y_B_pred)
# 
# data_test_modified_3 <- data_test %>%
#   mutate(MARS = y_mars, EN = y_EN, MLR = y_mlr, LASSO = y_LASSO, Poly = y_poly, Root = y_root, LOESS = y_loess, B = y_b)


data <- data_train %>%
  mutate(MARS = y_mars_pred, EN = y_EN_pred, MLR = y_mlr_pred, LASSO = y_LASSO_pred, Poly = y_poly_pred, Root = y_root_pred, LOESS = y_LOESS_pred)

data_test_modified_3 <- data_test %>%
  mutate(MARS = y_mars, EN = y_EN, MLR = y_mlr, LASSO = y_LASSO, Poly = y_poly, Root = y_root, LOESS = y_loess)
options(scipen = 999)


model_linear_inform <- lm(data = data, Balance ~ MARS + EN + MLR + LASSO + Poly + Root + LOESS + Root*MARS*MLR*LOESS*LASSO*Poly*EN) 
y_linear_inform <- predict(model_linear_inform, newdata = data_test_modified_3)
# summary(model_linear_inform)
# View(non_negative_model(y_linear_inform))


# model_linear_inform_2 <- lm(data = data, Balance ~ MARS + EN + MLR + LASSO + Poly + Root + LOESS + B + Root*MARS*Poly*MLR + B*LASSO*EN*MARS) 
# y_linear_inform_2 <- predict(model_linear_inform_2, newdata = data_test_modified_3)
# # sum(y_linear_inform_2 < 0)
# # summary(model_linear_inform_2)
# # View(non_negative_model(y_linear_inform_2))


model_quadratic_inform <- lm(data = data, Balance ~ poly(MARS,2,raw = T) + poly(EN,2,raw = T) + poly(MLR,2,raw = T) + poly(LASSO,2,raw = T) + poly(LOESS,2,raw = T) + poly(Poly,2,raw = T) + poly(Root,2,raw = T) +  Root*MARS*Poly*MLR*LOESS*LASSO*EN) 
y_quadratic_inform <- predict(model_quadratic_inform, newdata = data_test_modified_3)
# summary(model_quadratic_inform)
# View(non_negative_model(y_quadratic_inform))




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9 : MARS 2

data_2 <- data %>% mutate(Linear = model_linear_inform$fitted.values, Quadratic = model_quadratic_inform$fitted.values) %>% dplyr::select(-Balance)
data_test_modified_4 <- data_test_modified_3 %>% mutate(Linear = y_linear_inform, Quadratic = y_quadratic_inform)
x_3 <- as.matrix(data_2[,])


parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))
set.seed(8885)
cv_mars_model_2 <- train(x = x_3, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars_2 <- predict(object = cv_mars_model_2$finalModel, newdata = data_test_modified_4) 
# summary(cv_mars_model_2$finalModel)
# View(non_negative_model(y_mars_2))




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 9 : MARS 2

data_2 <- data %>% mutate(Linear = model_linear_inform$fitted.values, Quadratic = model_quadratic_inform$fitted.values) %>% dplyr::select(-Balance)
data_test_modified_4 <- data_test_modified_3 %>% mutate(Linear = y_linear_inform, Quadratic = y_quadratic_inform)
x_3 <- as.matrix(data_2[,])


parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))
set.seed(8885)
cv_mars_model_2 <- train(x = x_3, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars_2 <- predict(object = cv_mars_model_2$finalModel, newdata = data_test_modified_4) 
# summary(cv_mars_model_2$finalModel)
# View(non_negative_model(y_mars_2))




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 10: Inform Poly 2

y_mars_pred_2 <- predict(object = cv_mars_model_2$finalModel, newdata = data_2)
data_3 <- data_2 %>% mutate(MARS_2 = y_mars_pred_2)
data_test_modified_5 <- data_test_modified_4 %>% mutate(MARS_2 = y_mars_2)


model_inform <- lm(data = data_3, Balance ~ I(Quadratic^3) + I(MARS_2^3) + Linear*Quadratic*MARS_2) 
y_inform <- predict(model_inform, newdata = data_test_modified_5)
# summary(model_inform)
# View(non_negative_model(y_inform))

shapiro.test(y_inform)
par(mfrow = c(2,2))
plot(model_inform)


model_inform_transformed <- lm(data = data_3, sqrt(Balance) ~ sqrt(Linear - min(y_linear_inform)) + sqrt(Quadratic - min(y_quadratic_inform)) + sqrt(MARS_2 - min(y_mars_2)) + Linear*Quadratic*MARS_2) 
y_inform_transformed <- (predict(model_inform_transformed, newdata = data_test_modified_5))^2
# summary(model_inform_transformed)
# View(non_negative_model(y_inform_transformed))

shapiro.test(y_inform_transformed)
par(mfrow = c(2,2))
plot(model_inform_transformed)




