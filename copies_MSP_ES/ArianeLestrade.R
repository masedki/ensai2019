
########1. Chargement des donn?es ################

# J'ai pr?f?r? utilis? la base de donn?es indiqu?es dans vos derniers mails afin d'?tre s?re que tout 
#corresponde bien. 

# r?pertoire 
setwd("P:/3emeAnnee/apprentissage/projet")

# packages 
require(keras)
require(caret)
require (rlist)  # ? installer si n?cessaire 
library(doParallel)
require(kernlab)
library(tidyverse)
library(tibble)

#Chargement des donn?es 
mnist <-list.load('mnist2.rdata')
#typeof(mnist)
x_train<-mnist$train$x
y_train<-mnist$train$y
x_test<-mnist$test$x
y_test<-mnist$test$y

is.data.frame(x_test)

# transformation en dataframe
y_train<-data.frame(y_train)
x_train<-data.frame(x_train)
x_test<-data.frame(x_test)
y_test<-data.frame(y_test)

####################################################
########2. Pr?paration des donn?es ################

########QST 1. Aplatissement des images en vecteurs#####################################################
dim(x_test)
names(x_test)
# l'objet x_test a 10 000 observation et 784 variables qui correspondant aux 784 pixels 
# (qui ont des labels X7 ... X784)


dim(x_train)
names(x_test)
# l'objet x_train a 60 000 observation et toujours ces m?mes 784 variables. 

# la transformation en dataframe a applati mon jeu de donn?es. On aurait pu faire 
# un arrayshape.Par exemple :  
#x_train <- array_reshape(x_train, c(nrow(x_train), 784))
# x_test <- array_reshape(x_test, c(nrow(x_test), 784))


########QST 2. Suppression des pixels nuls pour l'ensemble des images du jeu de donn?es #########


x<-nearZeroVar(rbind(x_train,x_test),saveMetrics=TRUE, names=TRUE) #  ... 1
x <- data.frame(x)
x<-as_tibble(x)
#  table(x$nzv) # les nearZero pixels : 250 false, 534 true 
#table(x$zeroVar) # les zero pixel : 719 false 65 true
#???print(which(x$nzv==TRUE)) # la position des 534 pixels avec une variance quasi nulle (?chantillon apprentissage)
x_train2<-x_train[, -which(x$nzv==TRUE)]
x_test2<-x_test[,-which(x$nzv==TRUE)]
print(dim(x_train2)); print(dim(x_test2)) #250 pixels restants 


########QST 3. Procédure de réduction de dimension : fonction preProcess du package caret ######### ........... 1.5 

?preProcess

## Méthode Principal Component Analysis : 

# load the libraries
library(mlbench)
# calculate the pre-process parameters from the dataset
preprocessParamPCA <- preProcess(rbind(x_train2, x_test2), method=c("center", "scale", "pca"))# ...  0.5
# summarize transform parameters
print(preprocessParamPCA)
#Pre-processing:
#  - centered (250)
#- ignored (0)
#- principal component signal extraction (250)
#- scaled (250)
#PCA needed 90 components to capture 95 percent of the variance
# transformer le dataset apprentissage en utilisant les param?tres
x_train_pca <- predict(preprocessParamPCA, x_train2) # ...................   0.5
# transformer le dataset test en utilisant les param?tres
x_test_pca <- predict(preprocessParamPCA, x_test2) # ........................ 0.5
dim(x_test_pca) ; dim(x_train_pca) 
#90 var explicatives conserv?es pour l'?chantillon test et apprentissage


####################################################
########3. Apprentissage par SVM linéaire ################................................................ 1.5



# Ajouter la variable r?ponse ? la base de donn?es x_train et x_test : 
train_final<-cbind(x_train_pca,y_train)
dim(train_final)

test_final<-cbind(x_test_pca,y_test)
dim(test_final)
#train_final$y_train
#test_final$y_test



# Transforme en factor la variable r?ponse
#is.factor(train_final$y_train)
train_final$y_train <- as.factor(train_final$y_train)# ............................  0.5
test_final$y_test<-as.factor(test_final$y_test)




# Fixer les paramètres: C = 10**(-3:0), Validation croisée ? 5 folds répétée 2 fois
control=trainControl(method="repeatedcv", number=5, repeats=2)
grid= expand.grid(C=10**(-3:0))


# ... 1/2.5  pas de résultats !!!!

svm.train_final=train(y_train~.,data=train_final, 
                      method="svmLinear", 
                      trControl=control, 
                      tuneGrid=grid, 
                      metric="Accuracy")
# trouver le taux de bon classement : je n'arrive pas ? faire tourner cette
# ligne de code avec la VM pas assez puissante pour cela. 

# Si j'avais pu faire tourner mon code, j'aurais voulu tester ces lignes de code.
# Make predictions on the test data
predicted <- model %>% predict(test_final) ## ?????????????? 
head(predicted)
# Compute model accuracy rate
mean(predicted == test_final$y_test)


####5. implémentation d'un RNA avec une seule couche cachée avec keras#############......................................... 5.5

#mnist <-list.load('mnist2.rdata')

x_train<-mnist$train$x
y_train<-mnist$train$y
x_test<-mnist$test$x
y_test<-mnist$test$y

#is.data.frame(x_test)

# rescale :
x_train.rna=array_reshape(x_train,c(nrow(x_train),784))
x_test.rna = array_reshape(x_test,c(nrow(x_test),784))

x_train.rna<-x_train.rna/255
x_test.rna<-x_test.rna/255

y_train.rna <- to_categorical(y_train, 10)
y_test.rna <- to_categorical(y_test, 10)


#dim(x_test.rna)
#print(x_train.rna[,1])
#is.data.frame(x_test.rna)


#.................................... 1/1

modele.rna <- keras_model_sequential() 
modele.rna %>% 
  layer_dense(units = 784, activation = "relu", input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = "softmax")


# .................................. 4.5/4.5
modele.rna %>% compile(
  optimizer = optimizer_adam(),
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)
modele.rna %>% fit(x_train.rna, y_train.rna, validation_data=list(x_test.rna, y_test.rna), epochs=10, batch_size=200)
?fit
?keras_model_sequential()

#Epoch 10/10
#60000/60000 [==============================] - 17s - loss: 0.0089 - acc: 0.9982 - val_loss: 0.0613 - val_acc: 0.9823



scores <- modele.rna %>% evaluate(x_test.rna, y_test.rna,verbose = 0)
modele.rna %>% predict_classes(x_test.rna)

# Output metrics
cat('Test loss:', scores[[1]], '\n') #Test loss: 0.06129991
cat('Test accuracy:', scores[[2]], '\n') #Test accuracy: 0.9823 

# Le meilleur cycle d'apprentissage correspond au 10?me cycle avec un taux de bon classement de 98.23%. 



####7. implémentation d'un réseau de neurones convolutifs############# .................................................... 5.75
mnist <-list.load('mnist2.rdata')

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

dim(x_test)
# Input image dimensions
img_rows <- 28
img_cols <- 28

input_shape <- c(img_rows, img_cols, 1)

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255


cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')

# Convert class vectors to binary class matrices
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Define model .................................... 9*0.25 = 2.25 
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 30, kernel_size = c(5,5), activation = 'relu',input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 15, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.30) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dense(units = 50, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)
# Train model
history_Cnn <-  model %>% fit(
  x_train, y_train,
  batch_size =200,
  epochs = 10,
  validation_split = 0.2
)

plot(history_Cnn)

scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# Output metrics
cat('Test loss:', scores[[1]], '\n') #0,0291
cat('Test accuracy:', scores[[2]], '\n')# 0,9902

# explication des couches 

#l'opération de convolution : 1ere étape d'un CNN sur les images de formation. 
# - Comme le jeu de données est constitué d'images ? 2 dimensions, nous 
# utilisons la convolution 2D. 
# - On  pose des filtres (ici 30, puis 15) pour trouver les caractéristiques de 
# chaque classe.
# - Nous découpons l'image en groupe de pixel (5*5). 
#cette couche va comparer ces groupes de pixels avec des filtres en utilisant 
# opération mathématique appelée convolution : Si le groupe de pixel correspond 
#au filtre cela va ếtre proche de 1 dans le cas contraire cela sera proche de-1. ........... 0.5/1 

# Ensuite, nous faisons du MaxPooling2D utilisé pour les opérations de 
# regroupement afin de réduire le nombre total de noeuds 
# pour les couches à venir. Nous gardons pour un groupe de pixel (3*3), 
# le pixel qui correspond au nombre maximal.   .............................................  1/1

# Ceci permet de de faire ensuite de nouveau une couche de convolution
# avec un nombre de filtres plus petit (15) sur des groupes de pixels plus petits 
# (3*3) sur un nombre de pixels plus restreints, ceux qui concentrent 
# l'information la plus importante d'après la couche précédente. 

# Nous re-faisons une opération de MaxPooling pour sélectionner pour un groupe 
# de pixel (2*2), le pixel qui correspond au nombre maximal. 

# Puis, nous avons fait une couche Flatten qui est utilisé pour l'aplatissement.  ........... 1/1
# Il s'agit d'un processus de conversion de tous les tableaux ? deux dimensions
# résultants en un seul vecteur linéaire continu et long.  

# Nous avons utilisé des fonctions d'activations "relu" afin 
# d'ajouter de la non-linéarité au réseau, sinon le réseau ne serait plus 
# capable de calculer qu'une fonction linéaire. L'objectif est qu'il apprenne 
# des fonctions plus complexes.                         ..................................... 0.25/0.5           

# Nous avons ensuite utilisé des couches cachées plus classiques (type dense)
# composées d'abord de 120 neurones, puis 50 neurones, avec une fonction 
# d'activation relu. 

# Finalement pour la dernière couche, nous avons utilis? "softmax" comme 
# fonction d'activation car on est dans le cas d'un probl?me de 
# classification de classes multiples et aussi afin d'avoir une somme de 
# probabilité égale ? 1 pour les classes. .....................................................0.25/0.5

# Dans notre modèle, nous avons également utilisé des couches Droupout.........................0.5/1 
# Le dropout est une technique de régularisation qui introduit un biais et 
# diminue la variance afin de minimiser le phénomène présent de l'overfitting. 
# Effectivement, des neurones vont être supprimés  chaque epoch pour mettre 
# du bruit et diminuer l'overfitting.                                 


