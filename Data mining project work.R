setwd ("C:\\Users\\rahmant\\Downloads")

# Import data set

library (Hmisc)

data_EANC = spss.get ("Processed_EANC.sav",
                      use.value.labels = TRUE)
# check missing value

library(mice)

md.pattern(data_EANC)
# Omit missing values.

cc = !is.na(data_EANC$HUSWORK) #Omit Missing value

data_final = data_EANC[cc,]
# Exclude weight variable.

mydata = data_final[, -2]
mydata$EANC <- as.factor(mydata$EANC)
levels(mydata$EANC) <- c("no", "yes")
mydata$MAGE25_29 <- ifelse(mydata$MAGE == "25-29", 1, 0)
mydata$MAGE30_49 <- ifelse(mydata$MAGE == "30-49", 1, 0)
mydata$V106Primary <- ifelse(mydata$V106 == "Primary", 1, 0)
mydata$V106Secondary <- ifelse(mydata$V106 == "Secondary", 1, 0)
mydata$V106Higher <- ifelse(mydata$V106 == "Higher", 1, 0)
mydata$HEDUPrimary <- ifelse(mydata$HEDU == "Primary", 1, 0)
mydata$HEDUSecondary <- ifelse(mydata$HEDU == "Secondary", 1, 0)
mydata$HEDUHigher <- ifelse(mydata$HEDU == "Higher", 1, 0)
mydata$wealthMiddle <- ifelse(mydata$wealth == "Middle", 1, 0)
mydata$wealthRich <- ifelse(mydata$wealth == "Rich", 1, 0)
mydata$RELIGIONNon_muslim <- ifelse(mydata$RELIGION == "Non-muslim", 1, 0)
mydata$IPUnintended <- ifelse(mydata$IP == "Unintended", 1, 0)
mydata$V714Yes <- ifelse(mydata$V714 == "Yes", 1, 0)
mydata$MMExposed <- ifelse(mydata$MM == "Exposed", 1, 0)
mydata$V025Rural <- ifelse(mydata$V025 == "Rural", 1, 0)
mydata$regionSouthern <- ifelse(mydata$region == "Southern", 1, 0)
mydata$regionCentral <- ifelse(mydata$region == "Central", 1, 0)
mydata$regionEastern <- ifelse(mydata$region == "Eastern", 1, 0)
mydata$HFDYes <- ifelse(mydata$HFD == "Yes", 1, 0)
mydata$MARRIAGE.greater_18 <- ifelse(mydata$MARRIAGE == ">=18", 1, 0)
mydata$V151Female <- ifelse(mydata$V151 == "Female", 1, 0)
mydata$BAGE.greater_18 <- ifelse(mydata$BAGE == ">=18", 1, 0)
mydata$HCDECHusband.alone <- ifelse(mydata$HCDEC == "Husband alone", 1, 0)
mydata$HCDECOther.person <- ifelse(mydata$HCDEC == "Other person", 1, 0)
mydata$HUSWORK.Not_working <- ifelse(mydata$HUSWORK == "Not working", 1, 0)

final.data<-mydata[,-c(2:17)]
## data partitioning
source("C:/Users/rahmant/Downloads/myfunctions.R")
RNGkind (sample.kind = "Rounding") 
set.seed(0)
##Partitioning the data into 80:20 partitions
p2 <- partition.2(final.data, 0.8)
training.data <- p2$data.train
test.data <- p2$data.test

##Create Full model
model.glm = glm(EANC~.,
                data = training.data,
                family = binomial(link='logit'))

summary (model.glm)
library(car)
vif(model.glm)
# prediction on test data
library(caret)
pred.prob <- predict(model.glm, test.data, type = "response")
pred.y.test <- ifelse(pred.prob > 0.5, "yes", "no") # using cutoff = 0.5
confusionMatrix(as.factor(pred.y.test), as.factor(test.data$EANC), 
                positive = "yes")

##Variable selection with different methods##

##Stepwise##
library(caret)

## K-fold Cross Validation
# value of K equal to 10 
set.seed(0)
train_control <- trainControl(method = "cv", 
                              number = 10) 

# Fit K-fold CV model  
step_kcv <- train(EANC ~ ., data = training.data, family = "binomial", 
                  method = "glmStepAIC", trControl = train_control) 
print(step_kcv)
step_kcv$finalModel

# Confusion matrix for test data
pred.prob.test <- predict(step_kcv$finalModel, test.data, type = "response")
pred.y.test <- ifelse(pred.prob.test > 0.5, "yes", "no") # using cutoff = 0.5
confusionMatrix(as.factor(pred.y.test), as.factor(test.data$EANC), 
                positive = "yes")


### Lasso regression ####

glmnet.lasso <- train(EANC ~ ., data = training.data, method = "glmnet",
                      family = "binomial", trControl = train_control, 
                      tuneGrid = expand.grid(alpha = 1,lambda = seq(0.001,0.1,by = 0.001)))

glmnet.lasso 
plot(glmnet.lasso)

# best parameter
glmnet.lasso$bestTune

# best coefficient
lasso.model <- coef(glmnet.lasso$finalModel, glmnet.lasso$bestTune$lambda)
lasso.model

# prediction on test data
pred.prob.lasso <- predict(glmnet.lasso, s = glmnet.lasso$bestTune, test.data, type = "prob")
pred.y.lasso <- ifelse(pred.prob.lasso[,2] > 0.5, "yes", "no") # using cutoff = 0.5
confusionMatrix(as.factor(pred.y.lasso), as.factor(test.data$EANC), 
                positive = "yes")


### Ridge regression ####

glmnet.ridge <- train(EANC ~ ., data = training.data, method = "glmnet",
                      family = "binomial", trControl = train_control, 
                      tuneGrid = expand.grid(alpha = 0,lambda = seq(0.001,0.1,by = 0.001)))
glmnet.ridge 
plot(glmnet.ridge)

# best parameter
glmnet.ridge$bestTune

# best coefficient
ridge.model <- coef(glmnet.ridge$finalModel, glmnet.ridge$bestTune$lambda)
ridge.model

# prediction on test data
pred.prob.ridge <- predict(glmnet.ridge, s = glmnet.ridge$bestTune, test.data, type = "prob")
pred.y.ridge <- ifelse(pred.prob.ridge[,2] > 0.5, "yes", "no") # using cutoff = 0.5
confusionMatrix(as.factor(pred.y.ridge), as.factor(test.data$EANC), 
                positive = "yes")




## cost complexity cross validation ##
library(caret)
library(rpart)
library(rpart.plot)
set.seed(0)
train_control <- trainControl(method="cv", number=10)
cv.ct <- train(EANC ~ . , data = training.data, method = "rpart",
               trControl = train_control, tuneLength = 10)

print(cv.ct)
plot(cv.ct)
##final model
cv.ct$finalModel
prp(cv.ct$finalModel, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = -10)

# variable importance
cv.ct$finalModel$variable.importance
summary(cv.ct$finalModel)

# get prediction on the test data
pred.test.prune = predict(cv.ct$finalModel, test.data, type = 'class')

# create confusion matrix
confusionMatrix(pred.test.prune, test.data$EANC, positive = "yes")



###### Bagging #####
####################################
library(caret)
set.seed(0)
modelLookup("treebag")
## specify nbagg to control the number of trees. default value is 25 
bag <- train(EANC ~ . , data = training.data, method = "treebag", nbagg = 50)
print(bag)
plot(varImp(bag))

##final model
bag$finalModel

# get prediction on the test data
pred.test.bag = predict(bag$finalModel, test.data, type = 'class')

# create confusion matrix
confusionMatrix(pred.test.bag, test.data$EANC, positive = "yes")



####################################
###### Random Forest ###############
####################################
library(caret)
library(randomForest)
set.seed(0)
modelLookup("rf")
rf <- train(EANC ~ . , data = training.data, method = "rf", tuneLength = 3)

print(rf)
plot(varImp(rf))

rf$finalModel

# get prediction on the test data
pred.test.rf = predict(rf$finalModel, test.data, type = 'class')

# create confusion matrix
confusionMatrix(pred.test.rf, test.data$EANC, positive = "yes")




####################################
###### Adaboost ####################
####################################
library(caret)
library(ada)
modelLookup("ada")
set.seed(0)
train_control <- trainControl(method="cv", number=10)
ada <- train(EANC ~ . , data = training.data, method = "ada",
             trControl = train_control, tuneLength = 3)

print(ada)
plot(varImp(ada))
##final model
ada$finalModel

# get prediction on the test data
pred.test.ada = predict(ada$finalModel, test.data)
# create confusion matrix
confusionMatrix(pred.test.ada, test.data$EANC, positive = "yes")


##Logistic regression with selected variables##
formula.new = EANC ~ V106Primary+V106Secondary+V106Higher+HEDUPrimary+HEDUSecondary+HEDUHigher+wealthMiddle+wealthRich+
  MMExposed+V025Rural+HFDYes+BAGE.greater_18

model.glm.new = glm(formula.new,
                data = training.data,
                family = binomial(link='logit'))

summary (model.glm.new)

#######################
#######################
##Lift charts######


pred.y.test <- ifelse(pred.prob > 0.5, "yes", "no") # using cutoff = 0.5

##Lift chart for stepwise
pred.y.test.new <- ifelse(pred.y.test=="yes",1,0)
test.data$EANC<-ifelse(test.data$EANC=="yes",1,0)
gain <- gains(test.data$EANC, pred.y.test.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart for logistic regression model", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")

##Lift chart for Lasso
pred.y.lasso.new <- ifelse(pred.y.lasso=="yes",1,0)
gain <- gains(test.data$EANC, pred.y.lasso.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart of Lasso model", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")

##Lift chart for Ridge regression
pred.y.ridge.new <- ifelse(pred.y.ridge=="yes",1,0)
gain <- gains(test.data$EANC, pred.y.ridge.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart for Ridge model", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")

##Lift chart for cost complexity cross validation
pred.test.prune.new <- ifelse(pred.test.prune=="yes",1,0)
gain <- gains(test.data$EANC, pred.test.prune.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart for cost complexity", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")

##Lift chart for Bagging
pred.test.bag.new <- ifelse(pred.test.bag=="yes",1,0)
gain <- gains(test.data$EANC, pred.test.bag.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart for Bagging", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")

##Lift chart for Random forest
pred.test.rf.new <- ifelse(pred.test.rf=="yes",1,0)
gain <- gains(test.data$EANC, pred.test.rf.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart for Random forest", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")

##Lift chart for Adaboost
pred.test.ada.new <- ifelse(pred.test.ada=="yes",1,0)
gain <- gains(test.data$EANC, pred.test.ada.new)
gain

# Plot Lift chart: Percent cumulative response
x <- c(0, gain$depth)
pred.y <- c(0, gain$cume.pct.of.total)
avg.y <- c(0, gain$depth/100)
plot(x, pred.y, main = "Cumulative Lift Chart for Adaboost", xlab = "deciles", 
     ylab = "Percent cumulative response", type = "l", col = "red", cex.lab = 1.5)
lines(x, avg.y, type = "l")
