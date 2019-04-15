# PROJET - APPRENTISSAGE ----

# le chargement des packages a été compliqué, les données utilisées ont principalement été les données du package dslabs, qui sont un peu modifiées
# j'ai réalisé plusieurs "save" et "load" durant le projet
  # faute de place pour l'envoi, je ne peux pas transmettre les RData par mail : ils ont étés commenté 
  # l'ensemble du script devrait être réutilisable tel quel : 
  # pour la partie 7, j'ai remplacé le chargement de votre rdata (renommé "mnist_2828.rdata") par la ligne de commande, qui devrait fonctionner sur votre poste

# 1. Le jeu de données MNIST ------------------

library(keras)
library(caret)
library(tidyverse)

# manière "correcte" de charger les données, mais abouti sur un fichier mnist contenant des listes dont je ne peux pas extraire les informations :
# install.packages("keras", dep=TRUE)
# require(keras)
# install_keras(tensorflow = "1.5.0")
# mnist <- dataset_mnist()



# Faute de télécharger le "bon" jeu de données, je suis passée par le fonction read_mnist() de dslabs :
# install.packages("dslabs")
require(dslabs)
mnist <- read_mnist()

# je sauve ce jeu de données "similaire" dans un RData "mnist.RData"
# save(mnist, file = "mnist.RData")
# load("mnist.RData")

# le jeu de données n'est pas exactement le même que celui proposé dans l'énoncé : les libellés ne sont pas les mêmes et surtout, les données sont déjà applaties :
  # x est une matrice avec 10 000/60 000 lignes (selon test/train) avec 784 colonnes. 1 ligne par image, 1 colonne par pixel
  # plutôt que d'avoir un tableau à 3 dimensions avec 60 000 images de 28*28 pixels au lieu des 784 alignés.

# par la suite, vous m'avez envoyé le jeu de données "original", je l'utilise dans la partie finale (7)
# ce "bon" jeu de données est enregistré sous "mnist_2828.RData"

# extraction des pixels (x) et de leur interprétation (y) pour le jeu de données train et test
  # (les noms des objets sont légérement différents dans le jeu de données mnist de dslabs)
x_train <- mnist$train$images
y_train <- mnist$train$labels
x_test <- mnist$test$images
y_test <- mnist$test$labels

# visualisation des images (un peu de travail est nécessaire, car mes données sont à plat)
par(mfcol=c(6,6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in 1:36) {
  im <- x_train[idx,]
  im2 <- array(dim = c(28,28))
  for (k in 1:28){  # nécessaire de redimensionner à la main pour afficher les images correctement
    im2[k,] <- im[((k-1)*28+1):((k-1)*28+28)] # on recrée l'image ligne par ligne, pour aboutir à une image 28*28
  }
  im2 <- t(apply(im2, 2, rev))
  image(1:28, 1:28, im2, col=gray((0:255)/255),
        xaxt='n', main=paste(y_train[idx]))
}
rm(im2, idx, im, k)


# 2.Préparation des données -----------------------------------------------------


### 1 ================================================================== .................................................... 1.5
# reshape : inutile car les données sont déjà applaties dans mon cas, mais pour le code :
# x_train <- array_reshape(x_train, c(nrow(x_train), 784)) ..................................0.5
# x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# on cherche les pixels nuls pour TOUTES les images
pixel_nul_train <- nearZeroVar(x_train, uniqueCut = 0) #..................................... 0 : attention dangereux
pixel_nul_test <- nearZeroVar(x_test, uniqueCut = 0)

# vérification pour s'assurer que l'argument "uniqueCut" sélectionne bien uniquement les variables de variances strictement égale à 0
# apply(x_train, 2, var) 
# apply(x_test, 2, var)

# on crée la liste avec les pixels nuls dans au moins un des 2 échantillons
# ça ne sert à rien de garder des pixels dans un échantillon et pas dans l'autre, d'où fusion des pixels nuls
pixel_nul_global <- c(pixel_nul_test, pixel_nul_train) %>% sort() %>% unique()

# on nomme les pixels, puis on enlève ces pixels nuls
colnames(x_train) <- 1:784
colnames(x_test) <- 1:784
x_train <- x_train[, -pixel_nul_global]
x_test <- x_test[, -pixel_nul_global]


### 2 =====================================================================
# pour la réduction de dimension, on peut utiliser le principe de l’Analyse en Composantes Principales (ACP)
  # en faisant une analogie avec une ligne par individu (image) et une colonne par variable (position du pixel)
  # chaque variable est alors quantitative et sa valeur est l'intensité du pixel.

# on va au préalable appliquer la fonction preProcess pour transformer (centrer-réduire) les échantillons
preProc  <- preProcess(x_train) # ...................................................................  où est l'ACP !!!!!!!!!!!!!!!!!
x_train <- predict(preProc, x_train)
x_test <- predict(preProc, x_test)

# dimensions avant réduction de dimension : 10000/60000 * 666
dim(x_test) # 666 variables
dim(x_train)

# sauvegarde des x_train, x_test avant réduction de dimensions 
# save(x_train, x_test, file = "x_train_test.RData")
# load("x_train_test.RData") 

# réduction de dimensions.......................................................................... 1/2 
require(FactoMineR)
test <- PCA(x_test, graph = FALSE)
train <- PCA(x_train, graph = FALSE)
train$eig[,1][train$eig[,1] > 5] # on a 25 axes qui gardent + de 5% de l'information sur l'échantillon train
test$eig[,1][test$eig[,1] > 5] # on en a 27 pour l'échantillon test
# si on conserve les 25 premiers axes, la fonction train est trop longue (a tourné 1h30 sans résultats...)

# on prend un nombre de dimensions + faible :
train$eig[,1][train$eig[,1] > 10] # on a 10 axes qui gardent + de 10% de l'information sur l'échantillon train
test$eig[,1][test$eig[,1] > 10] # on en a 10 aussi pour l'échantillon test
train <- PCA(x_train, graph = FALSE, ncp = 10)
test <- PCA(x_test, graph = FALSE, ncp = 10)

tab_train <- train$ind$coord %>% as.data.frame()
tab_test <- test$ind$coord %>% as.data.frame()

# dimensions finale : 10000/60000 * 10
dim(tab_test)
dim(tab_train)

# pour éviter de prendre trop de temps à relancer l'ACP, on sauvegarde ces résultats
# save(tab_test, tab_train, file = "train_test10.RData")

# 3 Apprentissage ---------------------

# load("train_test10.RData")

# on ajoute la variable de résultat (y)
tab_test$chiffre <- as.factor(y_test)
tab_train$chiffre <- as.factor(y_train)

library(kernlab)

set.seed(1)

### 3 ========== .......................................................................................................... 2.5 
###3 .a) 
grid <- expand.grid(C = 10**(-3:0))

### 3.b)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)

svm.caret <- train(chiffre ~ ., data = tab_train, 
                   trControl = control, tuneGrid = grid, method = "svmLinear")

# save(svm.caret, file = "svm.RData")

### 3.c) taux de bon classés
svm.pred <- predict(svm.caret, tab_test[-11]) %>% cbind(tab_test)
table(svm.pred$., svm.pred$chiffre)  # table de contigence 
table(svm.pred$., svm.pred$chiffre)  %>% prop.table() * 100  # table de contigence pondérée
err <- sum(svm.pred$. != svm.pred$chiffre)/nrow(svm.pred)
err
# l'erreur de prédiction en ayant conservé 10 dimensions est de 0,51 %... 
# en d'autre terme, on vise juste 49 fois sur 100, presque 1 fois sur 2.
# c'est supérieur au hasard (où on aurait bon 1 fois sur 10), mais reste assez faible...


# peut-on augmenter la précision en augmentant le nombre de dimensions (avec un temps de calcul raisonnable) ?
# on refait la procédure en conservant + de dimensions :
train$eig[,1][train$eig[,1] > 7.5] # on a 14 axes qui gardent + de 7.5% de l'information sur l'échantillon train
test$eig[,1][test$eig[,1] > 7.5] # on en a 15 pour l'échantillon test
train <- PCA(x_train, graph = FALSE, ncp = 14)
test <- PCA(x_test, graph = FALSE, ncp = 14)

tab_train <- train$ind$coord %>% as.data.frame()
tab_test <- test$ind$coord %>% as.data.frame()

# pour éviter de prendre trop de temps à relancer l'ACP, on sauvegarde ces résultats
# save(tab_test, tab_train, file = "train_test14.RData")
# load("train_test14.RData")

# on ajoute la variable de résultat (y)
tab_test$chiffre <- as.factor(y_test)
tab_train$chiffre <- as.factor(y_train)

set.seed(1)
grid <- expand.grid(C = 10**(-3:0))
control <- trainControl(method = "repeatedcv", number = 5, repeats = 2)
svm.caret <- train(chiffre ~ ., data = tab_train, 
                   trControl = control, tuneGrid = grid, method = "svmLinear")

### taux de bon classés avec 14 dimensions
svm.pred <- predict(svm.caret, tab_test[-15]) %>% cbind(tab_test)
table(svm.pred$., svm.pred$chiffre)  # table de contigence 
table(svm.pred$., svm.pred$chiffre)  %>% prop.table() * 100  # table de contigence pondérée
err <- sum(svm.pred$. != svm.pred$chiffre)/nrow(svm.pred)
err
# l'erreur de prédiction en ayant conservé 14 dimensions est de 0,57 %... 
# c'est moins bon qu'avec 10 dimensions. Certains chiffres (0 par ex) sont mieux prédits, d'autres moins bien (1)


# 5 RNA ----------------

# on recharge les données (j'ai aussi fait cette partie avec le jeu de données déjà applaties de dslabs)

mnist <- read_mnist()
# load("mnist.RData") # pour aller + vite

x_train <- mnist$train$images
y_train <- mnist$train$labels
x_test <- mnist$test$images
y_test <- mnist$test$labels


# transformation en matrice
y_train <- to_categorical(y_train, 10) 
y_test <- to_categorical(y_test, 10) 

# reshape : pas nécessaire dans mon cas
# x_train <- array_reshape(x_train, c(nrow(x_train), 784)) 
# x_test <- array_reshape(x_test, c(nrow(x_test), 784)) 

# rescale 
x_train <- x_train / 255 
x_test <- x_test / 255


### 4 ============ .......................................................................................... 5.5
### 4.a) 
model <- keras_model_sequential() %>%  
  layer_dense(units = 784, activation = 'relu', input_shape = c(784)) %>% # couche d'entrée précisant la taille des données en entrée
  layer_dense(units = 784, activation = 'relu') %>% # couche cachée comportant 784 LTU, comme précisé
  layer_dense(units = 10, activation = 'softmax') # 10 LTU en sortie

summary(model)
# on a besoin de 10 LTU sur la couche de sortie, car on souhaite obtenir une probabilité pour chaque chiffre de 0 à 9...................... 0.5
# La fonction d'activation est la fonction d'activation "softmax", qui transforme un vecteur de K nombres réels en K probabilités.......... 0.5


### 4.b) ............................................................................................................................4.5
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = 'accuracy'
)

# choix de la fonction de perte "entropie croisée" pour catégories
  # (pour une classification binaire, la minimisation de l'entropie croisée correspond à la maximisation la log-vraisemblance)
  # (pour une classification multiple, on voudrait idéalement minimiser l'erreur de classification, 
  # qui n'est pas "smooth" -dérivable ?- donc on se sert de l'entropie croisée aussi) 
# choix de la métrique "accuracy" dans le cas d'une classification
# algorithme d'optimisation adam (Adaptative Moments)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, # cycles d'apprentissage
  batch_size = 200, # taille des échantillons
  validation_data = list(x_test, y_test) # inclure l'échantillon de test pour validation
)


# La fonction fit ne fonctionne pas et renvoi une erreur à propos d'un package Numpy
# Error in py_call_impl(callable, dots$args, dots$keywords) : 
# Evaluation error: Required version of NumPy not available: installation of Numpy >= 1.6 not found. 

# Une tentative pour installer Numpy dans la version 1.6 via Anaconda Prompt échoue :
# UnsatisfiableError: The following specifications were found to be in conflict:
#   - numpy=1.6
# - tensorflow=1.5 -> numpy[version='>=1.12.1']
# Use "conda search <package> --info" to see the dependencies for each package.

# On écrit donc le code sans avoir les résultats à partir de la fonction fit

# visualisation de la perte et précision du modèle
plot(history)

score <- model %>% evaluate(x_test, y_test)
paste('Perte du modèle :', score$loss, "\n")
paste('Précision du modèle :', score$acc, "\n")

# on ne peut indiquer la meilleure performance, faute de résultats


# 7 Réseau de neurones convolutifs ------------------------------------...................................................... 6

# obtention du fichier mnist en 3D (images en format 28*28 et non pas 784) que vous m'avez envoyé

# sur mon poste : utilisation du fichier que vous m'avez envoyé
# require (rlist)
# mnist <- list.load('mnist_2828.rdata')

mnist <- dataset_mnist() # cette ligne de commande ne me permettait pas de poursuivre, mais j'imagine qu'elle fonctionnera sur votre poste

x_train <- mnist$train$x 
y_train <- mnist$train$y 
x_test <- mnist$test$x 
y_test <- mnist$test$y

# normalisation des intensités 
x_train <- x_train / 255 
x_test <- x_test / 255


### 5 =============#.............................................................................................. 9 * 0.25
model <- keras_model_sequential() %>%   
    # 5.1 couche de convolution, appliquant des filtres de convolutions, "calque", transformant l'information de l'image  ........... 0.0
  layer_conv_2d(filters = 30, kernel_size = c(5,5), activation = 'relu', 
                input_shape = c(28, 28, 1)) %>% # image de taille 28*28, avec un seul paramétre de couleur (gris)
    # 5.2 couche max-pooling 2d,  réduisant la taille spatiale d'une image intermédiaire, ........................................... 1/1
    # en regroupant les tuiles de 2*2 pixels en un seul élément 
    # (et en affectant la valeur "max" des 4 pixels à la valeur de sortie)
    # ça réduit donc la quantité de paramètres et de calcul dans le réseau
    # le paramétre 2*2 permet d'éviter une perte trop importante d'information
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    # 5.3 couche de convulution, idem 5.1
  layer_conv_2d(filters = 15, kernel_size = c(3,3), activation = 'relu') %>%
    # 5.4 couche max-pooling 2d, idem 5.2
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
    # 5.5 couche dropout qui enlève des unités, afin d'éviter le surapprentissage en surintérprétant des relations dues au bruit....... 0.75/1
  layer_dropout(rate = 0.3) %>% 
    # 5.6 couche d'applatissement, qui transforme les tuiles (formes "carrées") en les verticalisant (forme "longue") ..................1/1
  layer_flatten() %>% 
    # 5.7 couche ReLU (Rectified linear unit), qui conserve les valeurs positives
    # permet d’accélérer et d’optimiser l’apprentissage..................................................................................0.5/1
  layer_dense(units = 128, activation = 'relu') %>%
    # 5.8 couche ReLU idem 5.7
  layer_dense(units = 50, activation = 'relu') %>% 
    # 5.9 comme précédemment, dernière couche softmax, qui permet d'obtenir nos probabilités sur les 10 chiffres ........................0.5/0.5
  layer_dense(units = 10, activation = 'softmax') %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_adam(),
          metrics = 'accuracy') 

summary(model)

history <- model %>% fit(x_train, y_train, 
      epochs = 10, # cycles d'apprentissage
      batch_size = 200, # taille des échantillons
      validation_data = list(x_test, y_test)) # inclure l'échantillon de test pour validation


# pour la même raison qu'à la partie précédente, on ne peut malheureusement pas faire tourner la fonction fit et donc obtenir des résultats