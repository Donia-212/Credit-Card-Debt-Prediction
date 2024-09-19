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

data_train_influential <- full_join(data_train, train_label, by = 'Index')
data_train <- full_join(data_train, train_label, by = 'Index')

ggplot(data_train, aes(x = Limit, y = Balance)) + 
  geom_point() + 
  facet_wrap(~ Region + Student)

ggplot(data_train, aes(x = Rating, y = Balance)) + 
  geom_point() + 
  facet_wrap(~ Region + Student)

ggplot(data_train, aes(x = Income, y = Balance)) + 
  geom_point() + 
  facet_wrap(~ Region)

ggplot(data_train, aes(x = Age ,y = Balance, color = Region)) +
  geom_boxplot(outlier.colour = "orange", outlier.size = 2) +
  facet_grid(~ Student + Married)

ggplot(data_train, aes(x = Age ,y = Rating, color = Region)) +
  geom_boxplot(outlier.colour = "orange", outlier.size = 2) +
  facet_grid(~ Student + Married)

ggplot(data_train, aes(x = Age ,y = Limit, color = Region)) +
  geom_boxplot(outlier.colour = "orange", outlier.size = 2) +
  facet_grid(~ Student + Married)
  # 98 year old balance 1999

ggplot(data_train, aes(x = Student ,y = Balance, color = Region)) +
  geom_boxplot(outlier.colour = "orange", outlier.size = 2) 



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




# ---------------------------------------------------------------------------------------------------------

# Check for multicollinearity:

# dev.off()
d <- data_train %>%dplyr::select(-Balance)
corr_matrix <-  round(cor(d), 2)
ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower",lab = TRUE) 
model <- lm(data = data_train, Balance ~ .)

# A high rating qualify for a higher credit limit
vif(model)



# ----------------------------------------------------------------------------------------------------------------------------------


# Model 1: Mars Model: 

x_train <- as.matrix(data_train[c('Income','Limit','Rating','Cards','Age','Student')])
x <- cbind(x_train, Income_Limit = c(data_train$Income*data_train$Limit), Income_Rating = c(data_train$Income*data_train$Rating), Rating_Limit = 
             c(data_train$Rating*data_train$Limit), Age_Limit = c(data_train$Age*data_train$Limit),Student_Limit = c(data_train$Student*data_train$Limit))

data_test_modified <- data_test %>% mutate(Income_Limit = data_test$Income*data_test$Limit, Income_Rating = data_test$Income*data_test$Rating,
                                           Rating_Limit = data_test$Rating*data_test$Limit, Age_Limit = data_test$Age*data_test$Limit,Student_Limit = 
                                             data_test$Student*data_test$Limit) %>% dplyr::select(-Own,-Married,-Region,-Education) 
y <- data_train$Balance
x_test <- as.matrix(data_test_modified[,])

parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))

set.seed(8885)
cv_mars_model <- train(x = x, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars <- predict(object = cv_mars_model$finalModel, newdata = data_test_modified) 
# View(non_negative_model(y_mars))
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

# Model 9: Ridge Model

model_Ridge <- cv.glmnet(x_2, y, alpha = 0, seed = 8885)
best_lambda_Ridge <- model_Ridge$lambda.min
model_Ridge <- glmnet(x_2, y, alpha = 0, lambda = best_lambda_Ridge)

y_Ridge <- predict(model_Ridge, newx = x_test_2)
# View(non_negative_model(y_Ridge))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 10: Bayesian

model_b <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student + Income:Limit + Income:Rating + Rating:Limit + Age:Limit + Student:Limit + Age:Student + Student:Income + Student:Rating + Student:Rating:Limit + Student:Rating:Limit:Income, seed = 8885)
y_b <- predict(model_b, newdata = data_test)
# View(non_negative_model(y_b))
# shapiro.test(resid(model_b))
# y_b_resid <- data_train$Balance - model_b$fitted.values
# qqnorm(y_b_resid)
# qqline(y_b_resid)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 11: Inform Linear

y_mars_pred <- predict(object = cv_mars_model$finalModel, newx = x)
y_EN_pred <- predict(model_EN, newx = x_2)
y_mlr_pred <- model_mlr$fitted.values
y_LASSO_pred <- predict(model_LASSO, newx = x_2)
y_poly_pred <- model_poly$fitted.values
y_root_pred <- model_root$fitted.values
y_LOESS_pred <- predict(object = cv_loess_model$finalModel, newx = x_2)
y_robust_pred <- model_robust$fitted.values
y_Ridge_pred <- predict(model_Ridge, newx = x_2)
y_B_pred <- model_b$fitted.values


data <- data_train %>%
  mutate(MARS = y_mars_pred, EN = y_EN_pred, MLR = y_mlr_pred, LASSO = y_LASSO_pred, Poly = y_poly_pred, Root = y_root_pred, LOESS = y_LOESS_pred, Robust = y_robust_pred, Ridge = y_Ridge_pred, B = y_B_pred)

data_test_modified_3 <- data_test %>%
  mutate(MARS = y_mars, EN = y_EN, MLR = y_mlr, LASSO = y_LASSO, Poly = y_poly, Root = y_root, LOESS = y_loess, Robust = y_robust, Ridge = y_Ridge, B = y_b)


options(scipen = 999)
model_linear_inform <- lm(data = data, Balance ~ Root*Poly*MLR + Ridge*MARS*LASSO*Poly*Robust*B + Ridge*LOESS*EN*MARS*Root*B + Ridge*LOESS*MLR*Root*EN*B + Root*MARS*MLR) 
y_linear_inform <- predict(model_linear_inform, newdata = data_test_modified_3)
# View(non_negative_model(y_linear_inform))
# shapiro.test(rstandard(model_linear_inform))
# par(mfrow = c(2,2)); plot(model_linear_inform)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 12: Inform Polynomial

model_poly_inform <- lm(data = data, Balance ~ I(MARS^2) + I(EN^2) + I(MLR^2) + I(LASSO^2) + I(LOESS^2) + I(Poly^2) + I(Root^2) + I(Robust^2) + I(B^2) +  Root*Poly*MLR + Ridge*MARS*LASSO*Poly*Robust*B + Ridge*LOESS*EN*MARS*Root*B + Ridge*LOESS*MLR*Root*EN*B + Root*MARS*MLR) 
y_poly_inform <- predict(model_poly_inform, newdata = data_test_modified_3)
# View(non_negative_model(y_poly_inform))
# shapiro.test(rstandard(model_poly_inform))
# par(mfrow = c(1,1)); plot(model_poly_inform, which = c(1,1))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 13: Inform Root


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
y_Ridge_pred_2 <- non_negative_model_2(predict(model_Ridge, newx = x_2))
y_B_pred_2 <- non_negative_model_2(model_b$fitted.values)


data_2 <- data_train %>%
  mutate(MARS = y_mars_pred_2, EN = y_EN_pred_2, MLR = y_mlr_pred_2, LASSO = y_LASSO_pred_2, Poly = y_poly_pred_2, Root = y_root_pred_2, LOESS = y_LOESS_pred_2, Robust = y_robust_pred_2, Ridge = y_Ridge_pred_2, B = y_B_pred_2)

data_test_modified_4 <- data_test %>%
  mutate(MARS = non_negative_model_2(y_mars), EN = non_negative_model_2(y_EN), MLR = non_negative_model_2(y_mlr), LASSO = non_negative_model_2(y_LASSO), Poly = non_negative_model_2(y_poly), Root = non_negative_model_2(y_root), LOESS = non_negative_model_2(y_loess), Robust = non_negative_model_2(y_robust), Ridge = non_negative_model_2(y_Ridge), B = non_negative_model_2(y_b))


model_root_inform <- lm(data = data_2, sqrt(Balance) ~ sqrt(MARS) + sqrt(EN) + sqrt(MLR) + sqrt(LASSO) + sqrt(LOESS) + sqrt(Poly) + Root + sqrt(Robust) + sqrt(Ridge) + sqrt(B) + Root*Poly*MLR + Ridge*MARS*LASSO*Poly*Robust*B + LOESS*EN*MARS*Root*B + LOESS*MLR*Root*EN*B + Root*MARS*MLR)
y_root_inform <- (predict(model_root_inform, newdata = data_test_modified_4))^2
# View(as_tibble(y_root_inform))
# shapiro.test(rstandard(model_root_inform))   
# par(mfrow = c(2,2)); plot(model_root_inform)



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 14: Inform Linear

model_linear_transform <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B)*sqrt(Ridge) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_linear_transform <- (predict(model_linear_transform, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform))
# shapiro.test(rstandard(model_linear_transform))
# par(mfrow = c(2,2)); plot(model_linear_transform)

data_3 <- data_2 %>% mutate(Balance = Balance + 1.38)
model_boxcox <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B)*sqrt(Ridge) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_boxcox <- (predict(model_boxcox, newdata = data_test_modified_4))^2

bc <- boxcox(model_boxcox)
lambda <- bc$x[which.max(bc$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox <- lm(data = data_3, ((sqrt(Balance)^lambda - 1)/lambda) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B)*sqrt(Ridge) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*sqrt(Root) + sqrt(LOESS)*sqrt(MLR)*sqrt(Root)*sqrt(EN) + sqrt(Root)*sqrt(MARS)*sqrt(MLR)) 
y_linear_transformed_boxcox <- (lambda * predict(model_linear_transformed_boxcox, newdata = data_test_modified_4) + 1) ^ (2/ lambda) 
y_linear_transformed_boxcox <- replace(y_linear_transformed_boxcox, is.nan(y_linear_transformed_boxcox), 0)
# View(as_tibble(y_linear_transformed_boxcox))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox)





model_linear_transform_2 <- lm(data = data_2, sqrt(Balance) ~ Root*sqrt(Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B)*sqrt(Ridge) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*Root + sqrt(LOESS)*sqrt(MLR)*Root*sqrt(EN) + Root*sqrt(MARS)*sqrt(MLR)) 
y_linear_transform_2 <- (predict(model_linear_transform_2, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_2))
# shapiro.test(rstandard(model_linear_transform_2))
# par(mfrow = c(2,2)); plot(model_linear_transform_2)


model_boxcox_2 <- lm(data = data_3, sqrt(Balance) ~ Root*sqrt(Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B)*sqrt(Ridge) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*Root + sqrt(LOESS)*sqrt(MLR)*Root*sqrt(EN) + Root*sqrt(MARS)*sqrt(MLR)) 
y_boxcox_2 <- (predict(model_boxcox_2, newdata = data_test_modified_4))^2

bc_3 <- boxcox(model_boxcox_2)
lambda_3 <- bc_3$x[which.max(bc_3$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_2 <- lm(data = data_3, ((sqrt(Balance)^lambda_3 - 1)/lambda_3) ~ Root*sqrt(Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Poly)*sqrt(Robust)*sqrt(B)*sqrt(Ridge) + sqrt(LOESS)*sqrt(EN)*sqrt(MARS)*Root + sqrt(LOESS)*sqrt(MLR)*Root*sqrt(EN) + Root*sqrt(MARS)*sqrt(MLR)) 
y_linear_transformed_boxcox_2 <- (lambda_3 * predict(model_linear_transformed_boxcox_2, newdata = data_test_modified_4) + 1) ^ (2/ lambda_3) 
y_linear_transformed_boxcox_2 <- replace(y_linear_transformed_boxcox_2, is.nan(y_linear_transformed_boxcox_2), 0)
# View(as_tibble(y_linear_transformed_boxcox_2))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_2)





model_linear_transform_3 <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust) + sqrt(B)*sqrt(Ridge)*sqrt(LOESS)*sqrt(EN)) 
y_linear_transform_3 <- (predict(model_linear_transform_3, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_3))
# shapiro.test(rstandard(model_linear_transform_3))
# par(mfrow = c(2,2)); plot(model_linear_transform_3)

model_boxcox_3 <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust) + sqrt(B)*sqrt(Ridge)*sqrt(LOESS)*sqrt(EN)) 
y_boxcox_3 <- (predict(model_boxcox_3, newdata = data_test_modified_4))^2

bc_3 <- boxcox(model_boxcox_3)
lambda_3 <- bc_3$x[which.max(bc_3$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_3 <- lm(data = data_3, ((sqrt(Balance)^lambda_3 - 1)/lambda_3) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust) + sqrt(B)*sqrt(Ridge)*sqrt(LOESS)*sqrt(EN)) 
y_linear_transformed_boxcox_3 <- (lambda_3 * predict(model_linear_transformed_boxcox_3, newdata = data_test_modified_4) + 1) ^ (2/ lambda_3) 
y_linear_transformed_boxcox_3 <- replace(y_linear_transformed_boxcox_3, is.nan(y_linear_transformed_boxcox_3), 0)
# View(as_tibble(y_linear_transformed_boxcox_3))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_3)






model_linear_transform_4 <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transform_4 <- (predict(model_linear_transform_4, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_4))
# shapiro.test(rstandard(model_linear_transform_4))
# par(mfrow = c(2,2)); plot(model_linear_transform_4)

model_boxcox_4 <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_boxcox_4 <- (predict(model_boxcox_4, newdata = data_test_modified_4))^2

bc_4 <- boxcox(model_boxcox_4)
lambda_4 <- bc_4$x[which.max(bc_4$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_4 <- lm(data = data_3, ((sqrt(Balance)^lambda_4 - 1)/lambda_4) ~ sqrt(Root*Poly*MLR) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transformed_boxcox_4 <- (lambda_4 * predict(model_linear_transformed_boxcox_4, newdata = data_test_modified_4) + 1) ^ (2/ lambda_4) 
y_linear_transformed_boxcox_4 <- replace(y_linear_transformed_boxcox_4, is.nan(y_linear_transformed_boxcox_4), 0)
# View(as_tibble(y_linear_transformed_boxcox_4))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_4)






model_linear_transform_5 <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(Robust)*sqrt(MARS)*sqrt(B) + sqrt(LOESS)*sqrt(Ridge)*sqrt(EN)*sqrt(LASSO)) 
y_linear_transform_5 <- (predict(model_linear_transform_5, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_5))
# shapiro.test(rstandard(model_linear_transform_5))
# par(mfrow = c(2,2)); plot(model_linear_transform_5)

model_boxcox_5 <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root*Poly*MLR) + sqrt(Robust)*sqrt(MARS)*sqrt(B) + sqrt(LOESS)*sqrt(Ridge)*sqrt(EN)*sqrt(LASSO)) 
y_boxcox_5 <- (predict(model_boxcox_5, newdata = data_test_modified_4))^2

bc_5 <- boxcox(model_boxcox_5)
lambda_5 <- bc_5$x[which.max(bc_5$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_5 <- lm(data = data_3, ((sqrt(Balance)^lambda_5 - 1)/lambda_5) ~ sqrt(Root*Poly*MLR) + sqrt(Robust)*sqrt(MARS)*sqrt(B) + sqrt(LOESS)*sqrt(Ridge)*sqrt(EN)*sqrt(LASSO)) 
y_linear_transformed_boxcox_5 <- (lambda_5 * predict(model_linear_transformed_boxcox_5, newdata = data_test_modified_4) + 1) ^ (2/ lambda_5) 
y_linear_transformed_boxcox_5 <- replace(y_linear_transformed_boxcox_5, is.nan(y_linear_transformed_boxcox_5), 0)
# View(as_tibble(y_linear_transformed_boxcox_5))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_5)



data_2_outliers <- data_2 %>% filter(Income != 28.941, Income != 20.791, Income != 20.103, Income != 20.089, Income != 92.112, Income != 13.676, Income != 24.050, Income != 27.590, Income != 15.629, Income != 10.363, Income != 30.733, Income != 28.575, Income != 20.150)
data_3_outliers <- data_3 %>% filter(Income != 28.941, Income != 20.791, Income != 20.103, Income != 20.089, Income != 92.112, Income != 13.676, Income != 24.050, Income != 27.590, Income != 15.629, Income != 10.363, Income != 30.733, Income != 28.575, Income != 20.150)


model_linear_transform_6 <- lm(data = data_2_outliers, sqrt(Balance) ~ sqrt(EN*Poly*MLR*Robust) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(Root)*sqrt(Robust)) 
y_linear_transform_6 <- (predict(model_linear_transform_6, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_6))
# shapiro.test(rstandard(model_linear_transform_6))
# par(mfrow = c(2,2)); plot(model_linear_transform_6)

model_boxcox_6 <- lm(data = data_3_outliers, sqrt(Balance) ~ sqrt(EN*Poly*MLR*Robust) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(Root)*sqrt(Robust)) 
y_boxcox_6 <- (predict(model_boxcox_6, newdata = data_test_modified_4))^2

bc_6 <- boxcox(model_boxcox_6)
lambda_6 <- bc_6$x[which.max(bc_6$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_6 <- lm(data = data_3_outliers, ((sqrt(Balance)^lambda_6 - 1)/lambda_6) ~ sqrt(EN*Poly*MLR*Robust) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(Root)*sqrt(Robust)) 
y_linear_transformed_boxcox_6 <- (lambda_6 * predict(model_linear_transformed_boxcox_6, newdata = data_test_modified_4) + 1) ^ (2/ lambda_6) 
y_linear_transformed_boxcox_6 <- replace(y_linear_transformed_boxcox_6, is.nan(y_linear_transformed_boxcox_6), 0)
# View(as_tibble(y_linear_transformed_boxcox_6))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_6)





model_linear_transform_7 <- lm(data = data_2, sqrt(Balance) ~ sqrt(MLR)*sqrt(Root*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transform_7 <- (predict(model_linear_transform_7, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_7))
# shapiro.test(rstandard(model_linear_transform_7))
# par(mfrow = c(2,2)); plot(model_linear_transform_7)

model_boxcox_7 <- lm(data = data_3, sqrt(Balance) ~ sqrt(MLR)*sqrt(Root*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_boxcox_7 <- (predict(model_boxcox_7, newdata = data_test_modified_4))^2

bc_7 <- boxcox(model_boxcox_7)
lambda_7 <- bc_7$x[which.max(bc_7$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_7 <- lm(data = data_3, ((sqrt(Balance)^lambda_7 - 1)/lambda_7) ~ sqrt(MLR)*sqrt(Root*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transformed_boxcox_7 <- (lambda_7 * predict(model_linear_transformed_boxcox_7, newdata = data_test_modified_4) + 1) ^ (2/ lambda_7) 
y_linear_transformed_boxcox_7 <- replace(y_linear_transformed_boxcox_7, is.nan(y_linear_transformed_boxcox_7), 0)
# View(as_tibble(y_linear_transformed_boxcox_7))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_7)





model_linear_transform_8 <- lm(data = data_2, sqrt(Balance) ~ sqrt(Root)*sqrt(MLR*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transform_8 <- (predict(model_linear_transform_8, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_8))
# shapiro.test(rstandard(model_linear_transform_8))
# par(mfrow = c(2,2)); plot(model_linear_transform_8)


model_boxcox_8 <- lm(data = data_3, sqrt(Balance) ~ sqrt(Root)*sqrt(MLR*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN))

y_boxcox_8 <- (predict(model_boxcox_8, newdata = data_test_modified_4))^2

bc_8 <- boxcox(model_boxcox_8)
lambda_8 <- bc_8$x[which.max(bc_8$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_8 <- lm(data = data_3, ((sqrt(Balance)^lambda_8 - 1)/lambda_8) ~ sqrt(Root)*sqrt(MLR*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transformed_boxcox_8 <- (lambda_8 * predict(model_linear_transformed_boxcox_8, newdata = data_test_modified_4) + 1) ^ (2/ lambda_8) 
y_linear_transformed_boxcox_8 <- replace(y_linear_transformed_boxcox_8, is.nan(y_linear_transformed_boxcox_8), 0)
# View(as_tibble(y_linear_transformed_boxcox_8))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_8)
# par(mfrow = c(2,1)); plot(model_linear_transformed_boxcox_8, which = c(1,2))
# summary(model_linear_transformed_boxcox_8)






data_2_outliers <- data_2 %>% filter(Income != 28.941, Income != 20.089, Income != 92.112, Income != 13.676, Income != 24.050, Income != 27.590, Income != 15.629, Income != 10.363, Income != 30.733, Income != 28.575, Income != 20.150)
data_3_outliers <- data_3 %>% filter(Income != 28.941, Income != 20.089, Income != 92.112, Income != 13.676, Income != 24.050, Income != 27.590, Income != 15.629, Income != 10.363, Income != 30.733, Income != 28.575, Income != 20.150)


model_linear_transform_9 <- lm(data = data_2_outliers, sqrt(Balance) ~ sqrt(MLR)*sqrt(Root*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transform_9 <- (predict(model_linear_transform_9, newdata = data_test_modified_4))^2
# View(non_negative_model(y_linear_transform_9))
# shapiro.test(rstandard(model_linear_transform_9))
# par(mfrow = c(2,2)); plot(model_linear_transform_9)


model_boxcox_9 <- lm(data = data_3_outliers, sqrt(Balance) ~ sqrt(MLR)*sqrt(Root*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_boxcox_9 <- (predict(model_boxcox_9, newdata = data_test_modified_4))^2

bc_9 <- boxcox(model_boxcox_9)
lambda_9 <- bc_9$x[which.max(bc_9$y)]

# Fit new model using the Box-Cox transformation
model_linear_transformed_boxcox_9 <- lm(data = data_3_outliers, ((sqrt(Balance)^lambda_9 - 1)/lambda_9) ~ sqrt(MLR)*sqrt(Root*Poly) + sqrt(MARS)*sqrt(LASSO)*sqrt(Robust)*sqrt(LOESS) + sqrt(B)*sqrt(Ridge)*sqrt(EN)) 
y_linear_transformed_boxcox_9 <- (lambda_9 * predict(model_linear_transformed_boxcox_9, newdata = data_test_modified_4) + 1) ^ (2/ lambda_9) 
y_linear_transformed_boxcox_9 <- replace(y_linear_transformed_boxcox_9, is.nan(y_linear_transformed_boxcox_9), 0)
# View(as_tibble(y_linear_transformed_boxcox_9))
# par(mfrow = c(2,2)); plot(model_linear_transformed_boxcox_9)




cbind(y_linear_transformed_boxcox_4, y_linear_transformed_boxcox_7, y_linear_transformed_boxcox_8, y_linear_transformed_boxcox_9)
cbind(glance(model_linear_transformed_boxcox_4), glance(model_linear_transformed_boxcox_7), glance(model_linear_transformed_boxcox_8), glance(model_linear_transformed_boxcox_9))
par(mfrow = c(4,2)); cbind(plot(model_linear_transformed_boxcox_4, which = c(1,2)), plot(model_linear_transformed_boxcox_7, which = c(1,2)),  plot(model_linear_transformed_boxcox_8, which = c(1,2)),  plot(model_linear_transformed_boxcox_9, which = c(1,2)))



# Inform Model

y_4 <- (model_linear_transform_4$fitted.values)^2
y_7 <- (model_linear_transform_7$fitted.values)^2
y_8 <- (model_linear_transform_8$fitted.values)^2


data_4 <- data_3 %>%
  mutate(Y4 = y_4, Y7 = y_7, Y8 = y_8)

data_5 <- data_3 %>%
  mutate(Y4 = y_4, Y7 = y_7, Y8 = y_8)

data_test_modified_5 <- data_test_modified_4 %>%
  mutate(Y4 = y_linear_transform_4, Y7 = y_linear_transform_7, Y8 = y_linear_transform_8)


model_Y <- lm(data = data_4, sqrt(Balance) ~ poly(Y4,2,raw = T)*poly(Y7,1,raw = T)*poly(Y8,1,raw = T)) 
Y <- predict(model_Y, newdata = data_test_modified_5)^2
# shapiro.test(resid(model_Y))
# par(mfrow = c(2,2)); plot(model_Y)


model_boxcox_Y <- lm(data = data_5, sqrt(Balance) ~ poly(Y4,2,raw = T)*poly(Y7,1,raw = T)*poly(Y8,1,raw = T)) 
Y_boxcox <- (predict(model_boxcox_Y, newdata = data_test_modified_5))^2

bc_Y <- boxcox(model_boxcox_Y)
lambda_Y <- bc_Y$x[which.max(bc_Y$y)]

# Fit new model using the Box-Cox transformation
model_transformed_boxcox_Y <- lm(data = data_5, ((sqrt(Balance)^lambda_Y - 1)/lambda_Y) ~ poly(Y4,2,raw = T)*poly(Y7,1,raw = T)*poly(Y8,1,raw = T)) 
y_transformed_boxcox_Y <- (lambda_Y * predict(model_transformed_boxcox_Y, newdata = data_test_modified_5) + 1) ^ (2 / lambda_Y) 
# View(as_tibble(y_transformed_boxcox_Y))
# par(mfrow = c(2,2)); plot(model_transformed_boxcox_Y)

cbind(glance(model_boxcox_Y), glance(model_linear_transformed_boxcox_4), glance(model_linear_transformed_boxcox_7), glance(model_linear_transformed_boxcox_8), glance(model_linear_transformed_boxcox_9))
cbind(y_transformed_boxcox_Y, y_linear_transformed_boxcox_4, y_linear_transformed_boxcox_7, y_linear_transformed_boxcox_8, y_linear_transformed_boxcox_9)




A <- ggplot(data = data_train) + 
  geom_point(aes(x = Income, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = (lambda_4 * model_linear_transformed_boxcox_4$fitted.values + 1)^(2/ lambda_4)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards', title = 'Model Linear Transformed BoxCox 4')


B <- ggplot(data = data_train) + 
  geom_point(aes(x = Limit, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = (lambda_4 * model_linear_transformed_boxcox_4$fitted.values + 1)^(2/ lambda_4)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards')

C <- ggplot(data = data_train) + 
  geom_point(aes(x = Rating, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = (lambda_4 * model_linear_transformed_boxcox_4$fitted.values + 1)^(2/ lambda_4)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards') 

ggarrange(A,B,C, ncol = 1, nrow = 3)




D <- ggplot(data = data_train) + 
  geom_point(aes(x = Income, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = (lambda_7 * model_linear_transformed_boxcox_7$fitted.values + 1)^(2/ lambda_7)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards', title = 'Model Linear Transformed BoxCox 7')


E <- ggplot(data = data_train) + 
  geom_point(aes(x = Limit, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = (lambda_7 * model_linear_transformed_boxcox_7$fitted.values + 1)^(2/ lambda_7)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards')

G <- ggplot(data = data_train) + 
  geom_point(aes(x = Rating, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = (lambda_7 * model_linear_transformed_boxcox_7$fitted.values + 1)^(2/ lambda_7)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards')

ggarrange(D,E,G, ncol = 1, nrow = 3)




H <- ggplot(data = data_train) + 
  geom_point(aes(x = Income, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = (lambda_8 * model_linear_transformed_boxcox_8$fitted.values + 1)^(2/ lambda_8)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards', title = 'Model Linear Transformed BoxCox 8')


I <- ggplot(data = data_train) + 
  geom_point(aes(x = Limit, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = (lambda_8 * model_linear_transformed_boxcox_8$fitted.values + 1)^(2/ lambda_8)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards', title = 'Model Linear Transformed BoxCox 8')

J <- ggplot(data = data_train) + 
  geom_point(aes(x = Rating, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = (lambda_8 * model_linear_transformed_boxcox_8$fitted.values + 1)^(2/ lambda_8)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards', title = 'Model Linear Transformed BoxCox 8') 

ggarrange(H,I,J, ncol = 1, nrow = 3)




data_train_outliers <- data_train %>% filter(Income != 28.941, Income != 20.089, Income != 92.112, Income != 13.676, Income != 24.050, Income != 27.590, Income != 15.629, Income != 10.363, Income != 30.733, Income != 28.575, Income != 20.150)

K <- ggplot(data = data_train_outliers) + 
  geom_point(aes(x = Income, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = (lambda_9 * model_linear_transformed_boxcox_9$fitted.values + 1)^(2/ lambda_9)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards', title = 'Model Linear Transformed BoxCox 9')


L <- ggplot(data = data_train_outliers) + 
  geom_point(aes(x = Limit, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = (lambda_9 * model_linear_transformed_boxcox_9$fitted.values + 1)^(2/ lambda_9)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards')

M <- ggplot(data = data_train_outliers) + 
  geom_point(aes(x = Rating, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = (lambda_9 * model_linear_transformed_boxcox_9$fitted.values + 1)^(2/ lambda_9)), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards') 

ggarrange(K,L,M, ncol = 1, nrow = 3)


ggarrange(A,D,H,K, ncol = 1, nrow = 4)
ggarrange(B,E,I,L, ncol = 1, nrow = 4)
ggarrange(C,G,J,M, ncol = 1, nrow = 4)



ggplot(data = data_train) + 
  geom_point(aes(x = data_train$Balance , y = (lambda_8 * model_linear_transformed_boxcox_8$fitted.values + 1)^(2/ lambda_8)), color = 'red') + 
  geom_smooth(aes(x = data_train$Balance, y = (lambda_8 * model_linear_transformed_boxcox_8$fitted.values + 1)^(2/ lambda_8)), se = F , color = 'forestgreen') + 
  labs(title = 'Model Linear Transformed BoxCox 8') +
  xlab('Balance') +
  ylab('Fitted Values')




# Checking for Influential points:

dfbetas <- as_tibble(dfbetas(model_linear_transformed_boxcox_4))
thresh <- 2/sqrt(350)

Q <- c()

for(i in seq(1,nrow(dfbetas))) {
  for(j in seq(1,length(dfbetas))){
    if(dfbetas[i,j] > thresh){
      #print(c(i,j))
      Q = c(Q,i)
    }
  }
}

Q <- unique(Q)
outliers <- data_train_influential %>% filter(Index %in% Q, Balance == 0 )
outliers



dfbetas <- as_tibble(dfbetas(model_linear_transformed_boxcox_7))
thresh <- 2/sqrt(350)

Q <- c()

for(i in seq(1,nrow(dfbetas))) {
  for(j in seq(1,length(dfbetas))){
    if(dfbetas[i,j] > thresh){
      #print(c(i,j))
      Q = c(Q,i)
    }
  }
}

Q <- unique(Q)
outliers <- data_train_influential %>% filter(Index %in% Q, Balance == 0 )
outliers




dfbetas <- as_tibble(dfbetas(model_linear_transformed_boxcox_8))
thresh <- 2/sqrt(350)

Q <- c()

for(i in seq(1,nrow(dfbetas))) {
  for(j in seq(1,length(dfbetas))){
    if(dfbetas[i,j] > thresh){
      #print(c(i,j))
      Q = c(Q,i)
    }
  }
}

Q <- unique(Q)
outliers <- data_train_influential %>% filter(Index %in% Q, Balance == 0 )
outliers



dfbetas <- as_tibble(dfbetas(model_linear_transformed_boxcox_9))
thresh <- 2/sqrt(350)

Q <- c()

for(i in seq(1,nrow(dfbetas))) {
  for(j in seq(1,length(dfbetas))){
    if(dfbetas[i,j] > thresh){
      #print(c(i,j))
      Q = c(Q,i)
    }
  }
}

Q <- unique(Q)
outliers <- data_train_influential %>% filter(Index %in% Q, Balance == 0 )
outliers



# Comparison Graph

# Generate some example data
set.seed(8885)
actual <- rnorm(50000, mean = 0, sd = 1)
predicted <- actual + rnorm(50000, mean = 0, sd = .35)  # Simulated predictions with noise

# Calculate residuals
residuals <- actual - predicted

# Define the loss functions
mse_loss <- residuals^2
mae_loss <- abs(residuals)
delta <- 1
huber_loss <- ifelse(abs(residuals) <= delta, 0.5 * (residuals^2), delta * (abs(residuals) - 0.5 * delta))

# Create a data frame for plotting
df <- data.frame(residuals = residuals, mse_loss = mse_loss, mae_loss = mae_loss, huber_loss = huber_loss)

# Plot with ggplot
ggplot(df, aes(x = residuals)) +
  geom_line(aes(y = mse_loss, color = "MSE"), size = 1) +
  geom_line(aes(y = mae_loss, color = "MAE"), size = 1) +
  geom_line(aes(y = huber_loss, color = "Huber Loss"), size = 1) +
  labs(x = "Residuals", y = "Loss", title = "Comparison Among Loss Functions") +
  scale_color_manual(values = c("darkred", "skyblue", "forestgreen"), 
                     labels = c("Huber Loss", "MAE", "MSE"))
