#Projet du cours "Apprentissage et aggégation de modèles" 

#MNIST ou le Hello World de la classification d'images

#On commence par acc?der au jeu de donn?es. 

rm(list=ls())

install.packages(keras,dep=TRUE)
require(keras)
require(caret)
require(e1071)
require(factoextra)
require(FactoMineR)

#On charge les donn?es
load("P:/MNIST/mnist.rdata")

#On applique une réduction de dimension ainsi
#que la fonction NearZeroVar pour supprimer les valeurs
#nulles

#Les données test n'étant pas censées ?tre disponibles, on applique 
#la procédure de train ? test. 

dim(x$train$x)<-c(nrow(x$train$x),784)
dim(x$test$x)<-c(nrow(x$test$x),784)

x_train<-x$train$x
x_test <- x$test$x

#nzv<-nearZeroVar(x_train)

#if(length(nzv) > 0) x_train <- x_train[, -nzv]
#if(length(nzv) > 0) x_test<- x_test[, -nzv]

#R?duction de dimension : on effectue une 
#ACP sur laquelle on ne conserve que les cinq 
#premiers axes

# On attribue des noms aux variables

colnames(x_train)<-c(1:783, "784")
colnames(x_test)<-c(1:783, "784")

## ................................................................................... 2 + 1
preprocessParams <- preProcess(x_train, method=c("nzv","center","scale","pca"))
print(preprocessParams)

transformed_train<-predict(preprocessParams, x_train)
transformed_test <- predict(preprocessParams,x_test)
summary(transformed_train)


# On associe les valeurs des coordonn?es aux valeurs de Y


transformed_train<- cbind(x$train$y,transformed_train)
transformed_test <- cbind(x$test$y, transformed_test)

# On fait tourner un SVM lin?aire avec crossvalidation ? 5 folds répétée deux fois........... 1/2.5


control=trainControl(method="repeatedcv",number=5, repeats = 2)
grid=expand.grid(C=10**(-3:0))
svm.x_train<-train(V1 ~., data=transformed_train,
                   trControl=control, tuneGrid=grid, 
                   method="svmLinear")

# Partie Réseau de Neurones artificiels ............................................... 1  +  2.0/4.5 aucun résultat 
# Implémentation d'un couche cachée ? 784 neurones. 

#La couche de sortie a pour fonction d'activation softmax, puisque  
#softmax permet de ne sélectionner qu'un seul neurone de la couche de 
#sortie sur les 10.
#On sélectionne d'ailleurs 10 parce qu'il y a 10 possibilités de sortie (de 0 ? 9)

modelSimple<-keras_model_sequential() %>% 
  layer_dense(units=784,input_shape=784,activation="relu") %>%
  layer_dense(units=10,activation="softmax")

# On minimise l'entropie de Shannon avec la fonction de perte binary_crossentropy.

modelMc %>% 
  compile(loss="binary_crossentropy",
          optimizer="adam",
          metrics="accuracy")


#processus d'apprentissage

modelMc %>% fit(x=XTrain,y=yTrain,epochs=200,batch_size=10)

get_weights(modelMc)

#fonction d'?valuation de Keras

res <-modelSimple %>% evaluate(x = x_test,y = y_test)

#affichage
print(res)

#Implémentation d'un réseau de neurones convolutifs .............  9*0.25  = 2.25

modelSimple<-keras_model_sequential() %>%
  layer_conv_2d(kernel=c(5,5), filters=30, activation="relu") %>%
  layer_global_max_pooling_2d(batch_size =c(2, 2), strides=None, padding='valid', data_format=None) %>%
  layer_conv_2d(kernel=c(3,3),filters=15, activation="relu") %>%
  layer_global_max_pooling_2d(batch_size =c(2, 2), strides=None, padding='valid', data_format=None) %>%
  layer_dropout(rate=0.3) %>%
  layer_flatten(data_format=None) %>%
  layer_dense(units=128,activation="relu") %>%
  layer_dense(units=50,activation="relu") %>%
  layer_dense(units=10,activation="softmax")
