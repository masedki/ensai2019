## 1.Le jeu de donn?es MNIST------

install.packages("keras", dep=TRUE)
#install.packages("caret")
#install.packages("MLmetrics")
#install.packages("e1071")
require(e1071)
require(MLmetrics)
require(keras)
require(rlist)
require(caret)
require(factoextra)
require(ModelMetrics)

setwd(dir ="P:/Cours/Master/apprentissage")
rm(list=ls())


mnist <- list.load("mnist.rdata")
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y


y_train <- factor(mnist$train$y,labels = c("Y0","Y1","Y2","Y3","Y4","Y5","Y6","Y7","Y8","Y9"))
y_test <- factor(mnist$test$y,labels = c("Y0","Y1","Y2","Y3","Y4","Y5","Y6","Y7","Y8","Y9"))

# visualize the digits
par(mfcol=c(6,6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in 1:36) {
  im <- x_train[idx,,]
  im <- t(apply(im, 2, rev))
  image(1:28, 1:28, im, col=gray((0:255)/255),
        xaxt='n', main=paste(y_train[idx]))
}

# je jette un oeil aux donn?es
str(x_train)
str(y_train)
summary(x_train)
summary(y_train)




##2.Préparation des données-----  1.5 + 0.75 = 2.25
#2.1 reshape
x_train <- keras::array_reshape(x_train, c(nrow(x_train), 784))  #.... 0.5
x_test <- keras::array_reshape(x_test, c(nrow(x_test), 784)) #.... 0.5 



# on supprime les colonnes n'apportant pas d'information
x_train_var_0 <- nearZeroVar(x_train) #  .... attention dangeureux : 0.5/1
x_train <- x_train[,-x_train_var_0]
x_test <- x_test[,-x_train_var_0]



#2.2 r?duction de dimensions : on peut faire une acp qui va r?duire les variables et enlever la colin?arit?

# on conserve 95% de la variance
preproc <- preProcess(data.frame(x_train), thresh = 0.95, method = "pca") # .... inutile de prendre tout : 0.75/1
x_train_pred <- predict(preproc, data.frame(x_train))
x_test_pred <- predict(preproc, data.frame(x_test))

# dans des data frame pour passer le svm lin?aire
train <- data.frame(x_train_pred,y_train)
test <- data.frame(x_test_pred,y_test)
# apr?s r?duction, il reste 90 composantes factorielles




##3. Apprentissage par SVM lin?aire----- # ... total = 2.5 


# Utiliser une validation crois?e ? 5 folds r?p?t?e deux fois2.
control <- caret::trainControl(method="repeatedcv", number = 5, repeats=2, summaryFunction=multiClassSummary, classProbs=TRUE) # ... 0.5

# Pour le param?tre C, on propose de tester la s?quence 10**(-3:0).
grid <- data.frame(C = 10**(-3:0)) # ... 0.5

modelFit <- train(y_train ~ . , data = train, method="svmLinear", metric="Accuracy", trControl=control, tuneGrid=grid) # ... 0.5

predictTest <- predict(object=modelFit,newdata = x_test_pred) # ... 0.5 

# Donner le taux de bon classement du jeu de donn?es test.
# 59 % d'images bien class?es ......................................0.5 
ratePredict <- mean(predictTest == y_test)



##4. Apprentissage par r?seaux de neurones artificiels-----


##5. Impl?mentation d'un RNA avec une seule couche cach?e avec keras-----   4.5 + 0.5 = 5  
require(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# a) on a besoin de 10 neurones (y train et test ont 10 modalit?s -> 1 neurone par modalit?) ... 0.5

# b)...  4.5 
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 784, activation = "relu", input_shape = c(784)) %>% 
      layer_dense(units = 10, activation = "softmax") # ... 0.5 

model %>% 
  compile(loss = "categorical_crossentropy", optimizer = optimizer_adam(), metrics = c("accuracy"))

history <- model %>% 
              fit(x_train, y_train, epochs = 10, batch_size = 200, validation_split = 0.2) # 

plot(history)# la meilleure performance est pour la 10?me it?ration 

history$metrics# 97.8 % de pr?cision pour cette 10?me it?ration

model %>% evaluate(x_test, y_test) # 98.06% de pr?cision sur l'ensemble du test




##6. Apprentissage par r?seaux de neurones convolutifs-----

##7. Impl?mentation d'un r?seau de neurones convolutifs-----

require(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

model <- keras_model_sequential()


model %>%  # 0 + (9*0.25) + 1 + 1 + 0.5  + 0.5  = 5.25
  # 1. la couche de convolution sort des images de dimension 28*28 sur une seule couleur avec un noyau de convolution de 5*5 ... NON
  layer_conv_2d(filters = 30, kernel_size = c(5,5), activation = "relu", input_shape = c(28,28,1)) %>% #... 0.25 
  # 2. la couche de maxpooling r?duit la dimension (ici de moiti? en largeur et longueur)#  ... 1 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% # ... 0.25   
  # avec 3 et 4 qui r?p?tent 1 et 2, on continue ? compresser l'image
  # 3. convolution
  layer_conv_2d(filters = 15, kernel_size = c(3,3), activation = "relu") %>% # ... 0.25 
  # 4. maxpooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>% # ... 0.25
  # 5. dropout pour éviter le surapprentissage  #... NON on peut expliquer le fonctionnement 
  layer_dropout(rate = 0.3) %>% # ... 0.25
  # 6. on applatit la matrice de pixels dans le même but qu'au début du tp # ... 1   
  layer_flatten() %>% # ... 0.25
  # 7. on fait un réseau de neurones pour terminer avec deux "relu" qui vont accéler la convergence ## ... 0.5/1
  layer_dense(units = 128, activation = "relu") %>% #... 0.25
  # 8.
  layer_dense(units = 50, activation = "relu") %>% #... 0.25
  # 9. et une logistique avec softmax pour finir  # ... 0.5
  layer_dense(units = 10, activation = "softmax") #... 0.25


model %>% compile(  loss = "categorical_crossentropy", optimizer = optimizer_adam(),  metrics = c("accuracy"))

x_train <- array(x_train, dim = c(60000, 28, 28, 1))  

history <- model %>% fit( x_train, y_train, epochs = 10, batch_size = 200, validation_split = 0.2)

plot(history)# la meilleure performance est pour la 10?me it?ration 

history$metrics# 99.1 % de pr?cision pour la 10?me it?ration


