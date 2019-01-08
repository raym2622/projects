e615 <- function(train_data,train_label,test_data,test_label,kernel,c){
  library(e1071)
train_data = read.table(train_data, header = F)
train_label = read.table(train_label, header = F)
test_data = read.table(test_data, header = F)
test_label = read.table(test_label, header = F)
train_data_scaled = scale(train_data)
test_data_scaled = scale(test_data)
train_label[train_label==0]=-1
test_label[test_label==0]=-1
svm.model <- svm(x=train_data_scaled, y=train_label,kernel=kernel, cost=c)
#linear kernel == "linear", Gaussian kernel == "radial"
svm.pred <- predict(svm.model, test_data_scaled)
pred = ifelse(svm.pred > 0, 1, -1)
table(pred,test_label[,1])
}

caret615 <- function(train_data,train_label,test_data,test_label,method){
  library(lattice)
  library(ggplot2)
  library(caret)
  train_data = read.table(train_data, header = F)
  train_label = read.table(train_label, header = F)
  test_data = read.table(test_data, header = F)
  test_label = read.table(test_label, header = F)
  #standardization will be done by preProcess in train().
  train_label[train_label==0]=-1
  test_label[test_label==0]=-1
  trctrl <- trainControl(method="repeatedcv",number=10,repeats=5)
  #method:resampling method;
  #number:number of resampling iteration;
  #repeats:number of complete sets of folds to compute.
  set.seed(1000)
  fit <- train(x=train_data,y=as.factor(train_label$V1),preProcess=c("center","scale"),trControl=trctrl,method=method,tuneLength=5)
  #method="svmLinear"(linear)/"svmRadial"(Gaussian)
  pred <- predict(fit, newdata = test_data)
  confusionMatrix(pred,as.factor(test_label$V1))
}

library(Smo615)
library(microbenchmark)
microbenchmark(smo615("sample_data.txt","sample_label.txt","test_data.txt","test_label.txt",100,0),e615("sample_data.txt","sample_label.txt","test_data.txt","test_label.txt","linear",100),caret615("sample_data.txt","sample_label.txt","test_data.txt","test_label.txt","svmLinear"))
microbenchmark(smo615("circle_train_data.txt","circle_train_label.txt","circle_test_data.txt","circle_test_label.txt",1,1),e615("circle_train_data.txt","circle_train_label.txt","circle_test_data.txt","circle_test_label.txt","radial",1),caret615("circle_train_data.txt","circle_train_label.txt","circle_test_data.txt","circle_test_label.txt","svmRadial"))