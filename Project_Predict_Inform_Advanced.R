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

x <- as.matrix(data_train[c('Income','Limit','Rating','Cards','Age','Student')])
y <- data_train$Balance

# Create a parameter tuning 'grid
parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))

# Cross Validation
set.seed(8885)
cv_mars_model <- train(x = x, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
#ggplot(cv_mars_model)
y_mars <- predict(object = cv_mars_model$finalModel, newdata = data_test) 
#View(non_negative_model(y_mars))


# ----------------------------------------------------------------------------------------------------------------------------------


# Model 2: EN Model

x_test <- as.matrix(data_test[c('Income','Limit','Rating','Cards','Age','Student')])

parameter_grid_en <- expand.grid(alpha = seq(0,1, length = 10), lambda = seq(0.0001,0.2, length = 5))

train_control <- trainControl(method = "repeatedcv",
                              number = 5,
                              repeats = 5,
                              search = "random",
                              verboseIter = TRUE)

set.seed(8885)
cv_en<- train(x,y, method = "glmnet", metric = 'Rsquared', trControl = train_control, tuneGrid = parameter_grid_en)
set.seed(8885)
model_EN <- glmnet(x, y, alpha = cv_en$bestTune[1], lambda = cv_en$bestTune[2])
y_en <- predict(model_EN, x_test)
#View(non_negative_model(y_en))




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 3: MLR Model
data_test_modified <- data_test %>% dplyr::select(1:5, Student)

model_mlr <- lm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student)
y_mlr <- predict(model_mlr, data_test_modified)
#View(non_negative_model(y_mlr))



# ----------------------------------------------------------------------------------------------------------------------------------

# Model 4: Bayesian Model

set.seed(8885)
model_b <- stan_glm(data = data_train, Balance ~ Income + Limit + Rating + Cards + Age + Student)
set.seed(8885)
c_b <- coefficients(model_b) 
y_b <- c_b[1] + c_b[2]*data_test$Income + c_b[3]*data_test$Limit + c_b[4]*data_test$Rating + c_b[5]*data_test$Cards + c_b[6]*data_test$Age + c_b[7]*data_test$Student
#View(non_negative_model(y_b))




# ----------------------------------------------------------------------------------------------------------------------------------


# Model 5: LASSO Model

set.seed(8885)
model_LASSO <- cv.glmnet(x, y, alpha = 1)
set.seed(8885)
best_lambda_LASSO <- model_LASSO$lambda.min
model_LASSO <- glmnet(x, y, alpha = 1, lambda = best_lambda_LASSO)

y_LASSO <- predict(model_LASSO, newx = x_test)
#View(non_negative_model(y_LASSO))




# ----------------------------------------------------------------------------------------------------------------------------------


# Model 6: Ridge Model

set.seed(8885)
model_Ridge <- cv.glmnet(x, y, alpha = 0)
set.seed(8885)
best_lambda_Ridge <- model_Ridge$lambda.min
model_Ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda_Ridge)

y_Ridge <- predict(model_Ridge, newx = x_test)
# View(non_negative_model(y_Ridge))



# ----------------------------------------------------------------------------------------------------------------------------------


# Model 5: Inform

y_mlr_pred <- model_mlr$fitted.values
y_EN_pred <- predict(model_EN, newx = x)
y_B_pred <- model_b$fitted.values
y_mars_pred <- predict(object = cv_mars_model$finalModel, newdata = data_train)
y_Ridge_pred <- predict(model_Ridge, newx = x)
y_LASSO_pred <- predict(model_LASSO, newx = x)


data <- data_train %>%
  mutate(MLR = y_mlr_pred, EN = y_EN_pred, B = y_B_pred, MARS = y_mars_pred, Ridge = y_Ridge_pred, LASSO = y_LASSO_pred)

data_test_modified <- data_test %>%
  mutate(MLR = y_mlr, EN = y_en, B = y_b, MARS = y_mars, Ridge = y_Ridge, LASSO = y_LASSO)




# Model Inform Linear:
options(scipen = 999)
model_poly_linear_inform <- lm(data = data, Balance ~ MLR + EN + B + MARS + LASSO +  Ridge + MLR*B*EN*MARS*Ridge*LASSO) 
y_poly_linear_inform <- predict(model_poly_linear_inform, newdata = data_test_modified)
# View(non_negative_model(y_poly_linear_inform))



# Model Inform Quadratic (Advanced):
model_poly_quadratic_inform <- lm(data = data, Balance ~ MLR + EN + B + MARS + LASSO +  Ridge + I(MLR^2) + I(EN^2) + I(B^2) + I(Ridge^2) + I(LASSO^2) + I(MARS^2) + MLR*B*EN*MARS*Ridge*LASSO + I(MARS^2)*I(B^2)*I(EN^2)*I(MLR^2)*I(Ridge^2)*I(LASSO^2)) 
y_poly_quadratic_inform <- predict(model_poly_quadratic_inform, newdata = data_test_modified)
# View(non_negative_model(y_poly_quadratic_inform))




# Model Inform Cubic:
# model_poly_cubic_inform <- lm(data = data, Balance ~ MLR + EN + B + MARS + LASSO +  Ridge + I(MLR^2) + I(EN^2) + I(B^2) + I(Ridge^2) + I(LASSO^2) + I(MARS^2) + I(MLR^3) + I(EN^3) + I(B^3) + I(Ridge^3) + I(LASSO^3) + I(MARS^3) + MLR*B*EN*MARS*Ridge*LASSO + I(MARS^2)*I(B^2)*I(EN^2)*I(MLR^2)*I(Ridge^2)*I(LASSO^2) + I(MLR^3)*I(EN^3)*I(B^3)*I(Ridge^3)*I(LASSO^3)*I(MARS^3) ) 
# y_poly_cubic_inform <- predict(model_poly_cubic_inform, newdata = data_test_modified)
# View(non_negative_model(y_poly_cubic_inform))
# 
# 
# 
# Model Inform Quartic:
# model_poly_quartic_inform <- lm(data = data, Balance ~ MLR + EN + B + MARS + LASSO +  Ridge + I(MLR^2) + I(EN^2) + I(B^2) + I(Ridge^2) + I(LASSO^2) + I(MARS^2) + I(MLR^3) + I(EN^3) + I(B^3) + I(Ridge^3) + I(LASSO^3) + I(MARS^3) + I(MLR^4) + I(EN^4) + I(B^4) + I(Ridge^4) + I(LASSO^4) + I(MARS^4) + MLR*B*EN*MARS*Ridge*LASSO + I(MARS^2)*I(B^2)*I(EN^2)*I(MLR^2)*I(Ridge^2)*I(LASSO^2) + I(MLR^3)*I(EN^3)*I(B^3)*I(Ridge^3)*I(LASSO^3)*I(MARS^3) + I(MLR^4)*I(EN^4)*I(B^4)*I(Ridge^4)*I(LASSO^4)*I(MARS^4)) 
# y_poly_quartic_inform <- predict(model_poly_quartic_inform, newdata = data_test_modified)
# View(non_negative_model(y_poly_quartic_inform))




# ----------------------------------------------------------------------------------------------------------------------------------


# Checking the assumptions:
options(scipen = 9)
par(mfrow = c(2,2))

# Adequate Models:
plot(model_poly_linear_inform)
shapiro.test(model_poly_linear_inform$residuals)$p.value

plot(model_poly_quadratic_inform)
shapiro.test(model_poly_quadratic_inform$residuals)$p.value


# Models are not adequate:
# plot(model_poly_cubic_inform)
# shapiro.test(model_poly_cubic_inform$residuals)$p.value
# 
# 
# plot(model_poly_quartic_inform)
# shapiro.test(model_poly_quartic_inform$residuals)$p.value


# Best Model is Inform Linear Model as it is the simplest and from the model diagnostic plots we 
# confirm that the assumptions of linear regression model are met. Also, the p value of shapiro 
# wilk test is 0.9349195



# ----------------------------------------------------------------------------------------------------------------------------------


A <- ggplot(data = data, aes(x = model_poly_linear_inform$residuals)) +
  geom_histogram(aes(y = ..density..), color = 'black', fill = 'white', bins = 30) +
  stat_function(fun = dnorm, args = list(mean = mean(model_poly_linear_inform$residuals), sd = sd(model_poly_linear_inform$residuals)), color = 'red', size = 1.5) +
  xlab('Residual') + 
  ylab('Density') +
  ggtitle('Normality of Linear Polynomial Model')


B <- ggplot(data = data, aes(x = model_poly_linear_inform$fitted.values, y = model_poly_linear_inform$residuals)) +
  geom_point() +
  geom_ref_line(h= 0, size = 1, colour = 'white') +
  xlab('Prediction') + 
  ylab('Residual') +
  ggtitle('Linear Polynomial : Residual vs Fitted')


C <- ggplot(data = data, aes(x = model_poly_quadratic_inform$residuals)) +
  geom_histogram(aes(y = ..density..), color = 'black', fill = 'white', bins = 50) +
  stat_function(fun = dnorm, args = list(mean = mean(model_poly_quadratic_inform$residuals), sd = sd(model_poly_quadratic_inform$residuals)), color = 'red', size = 1.5) +
  xlab('Residual') + 
  ylab('Density') +
  ggtitle('Normality of Quadratic Polynomial Model')


D <- ggplot(data = data, aes(x = model_poly_quadratic_inform$fitted.values, y = model_poly_quadratic_inform$residuals)) +
  geom_point() +
  geom_ref_line(h= 0, size = 1, colour = 'white') +
  xlab('Prediction') + 
  ylab('Residual') +
  ggtitle('Quadratic Polynomial : Residual vs Fitted')

ggarrange(A,B,C,D, ncol = 2, nrow = 2)



E <- ggplot(data = data_train) + 
  geom_point(aes(x = Income, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = model_poly_linear_inform$fitted.values), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards')

G <- ggplot(data = data_train) + 
  geom_point(aes(x = Limit, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = model_poly_linear_inform$fitted.values), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards')

H <- ggplot(data = data_train) + 
  geom_point(aes(x = Rating, y = Balance, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = model_poly_linear_inform$fitted.values), se = F) +
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards') 

ggarrange(E,G,H, ncol = 1, nrow = 3)



I <- ggplot(data = data_test) + 
  geom_point(aes(x = Income, y = y_poly_linear_inform, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Income, y = y_poly_linear_inform), se = F) + 
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards') +
  ylab('Predicted Values')

J <- ggplot(data = data_test) + 
  geom_point(aes(x = Limit, y = y_poly_linear_inform, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Limit, y = y_poly_linear_inform), se = F) + 
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards') +
  ylab('Predicted Values')

K <- ggplot(data = data_test) + 
  geom_point(aes(x = Rating, y = y_poly_linear_inform, colour = as.factor(Cards))) + 
  geom_smooth(aes(x = Rating, y = y_poly_linear_inform), se = F) + 
  facet_wrap(~Student, labeller = label_both) +
  labs(color = 'Number of Cards') +
  ylab('Predicted Values')

ggarrange(E,I, ncol = 1, nrow = 2)
ggarrange(G,J, ncol = 1, nrow = 2)
ggarrange(H,K, ncol = 1, nrow = 2)




# ----------------------------------------------------------------------------------------------------------------------------------


# Discard LASSO and Ridge

# Model Inform Linear without LASSO or Ridge:
model_poly_linear_inform_MEBM <- lm(data = data, Balance ~ MLR + EN + B + MARS + MLR*B*EN*MARS) 
y_poly_linear_inform_MEBM <- predict(model_poly_linear_inform_MEBM, newdata = data_test_modified)
# View(non_negative_model(y_poly_linear_inform_MEBM))

# Still a good model but the linear model with all 6 models is still better




# Discard Ridge

# Model Inform Linear without Ridge:
model_poly_linear_inform_MEBML <- lm(data = data, Balance ~ MLR + EN + B + MARS + LASSO + MLR*B*EN*MARS*LASSO) 
y_poly_linear_inform_MEBML <- predict(model_poly_linear_inform_MEBML, newdata = data_test_modified)
# View(non_negative_model(y_poly_linear_inform_MEBML))

# Still a good model but the linear model with all 6 models is still better


# ----------------------------------------------------------------------------------------------------------------------------------


# Model 6: LOESS

parameter_grid_loess <- expand.grid(span = seq(0.5, 0.9, len = 5), degree = 1)
set.seed(8885)
cv_model_loess <- train(x, y, method = "gamLoess", metric = 'Rsquared', tuneGrid = parameter_grid_loess, trControl = trainControl(method = "cv"))
set.seed(8885)
y_loess <- predict(cv_model_loess, data_test)
# View(non_negative_model(y_loess))




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 7 : MARS 2
data_train_2 <- data_train %>% mutate(MARS = predict(object = cv_mars_model$finalModel, newdata = data_train), B = model_b$fitted.values, LASSO = predict(model_LASSO, newx = x), MLR = model_mlr$fitted.values)
data_test_2 <- data_test %>% mutate(MARS = y_mars, B = y_b, LASSO = y_LASSO, MLR = y_mlr)
x_2 <- as.matrix(data_train_2[c('Income','Limit','Rating','Cards','Age','Student','MARS','B', 'LASSO', 'MLR')])

parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))
set.seed(8885)
cv_mars_model_2 <- train(x = x_2, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars_2 <- predict(object = cv_mars_model_2$finalModel, newdata = data_test_2) 
#View(non_negative_model(y_mars_2))

 
plot(cv_mars_model_2$finalModel)
shapiro.test(y_mars_2)



data_train_3 <- data_train %>% mutate(MARS = predict(object = cv_mars_model$finalModel, newdata = data_train), B = model_b$fitted.values, LASSO = predict(model_LASSO, newx = x), MLR = model_mlr$fitted.values, Poly = model_poly_linear_inform$fitted.values)
data_test_3 <- data_test %>% mutate(MARS = y_mars, B = y_b, LASSO = y_LASSO, MLR = y_mlr, Poly = y_poly_linear_inform)
x_3 <- as.matrix(data_train_3[c('Income','Limit','Rating','Cards','Age','Student','MARS','B', 'LASSO', 'MLR', 'Poly')])

parameter_grid <- floor(expand.grid(degree = 1:4, nprune = seq(5,50,5)))
set.seed(8885)
cv_mars_model_3 <- train(x = x_3, y = y, method = 'earth', metric = 'Rsquared', trControl = trainControl(method = 'cv'), tuneGrid = parameter_grid) 
set.seed(8885)
y_mars_3 <- predict(object = cv_mars_model_3$finalModel, newdata = data_test_3) 
#View(non_negative_model(y_mars_3))


plot(cv_mars_model_3$finalModel)
shapiro.test(y_mars_3)




# ----------------------------------------------------------------------------------------------------------------------------------

# Model 8: Inform Linear Poly 2

y_mars_pred_3 <- predict(object = cv_mars_model_3$finalModel, newdata = data_train_3)
y_poly_pred <- model_poly_linear_inform$fitted.values
y_Loess_pred <- predict(object = cv_model_loess$finalModel, newdata = data_train)

data_2 <- data %>%
  mutate(MARS_3 = y_mars_pred_3, Poly = y_poly_pred, Loess = y_Loess_pred)

data_test_modified_2 <- data_test_modified %>%
  mutate(MARS_3 = y_mars_3, Poly = y_poly_linear_inform, Loess = y_loess)

model_poly_linear_inform_2 <- lm(data = data_2, Balance ~ Poly + MARS_3 + I(MARS_3^2) + I(Poly^2) + MARS_3*Poly) 
y_poly_linear_inform_2 <- predict(model_poly_linear_inform_2, newdata = data_test_modified_2)
# View(non_negative_model(y_poly_linear_inform_2))

shapiro.test(y_poly_linear_inform_2)
par(mfrow = c(2,2))
plot(model_poly_linear_inform_2)



# Model 9: Inform Linear Poly 3

model_poly_linear_inform_3 <- lm(data = data_2, Balance ~ Poly + MARS_3 + Loess + I(MARS_3^2) + I(Poly^2) + I(Loess^2) + MARS_3*Poly*Loess) 
y_poly_linear_inform_3 <- predict(model_poly_linear_inform_3, newdata = data_test_modified_2)
# View(non_negative_model(y_poly_linear_inform_3))

shapiro.test(y_poly_linear_inform_3)
par(mfrow = c(2,2))
plot(model_poly_linear_inform_3)



