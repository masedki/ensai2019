# PROJET - APPRENTISSAGE # 
# 10 mars 2019 - Laurie Pinel 
# Le chargement des packages et des données a été compliqué et a entrainé le besion d'un délai supplémentaire
# J'ai du utiliser le jeu de données R.Data pour pouvoir démarer le projet 

# Préalable : instalation des packages # 

install.packages("rlist")
install.packages("keras", dep=TRUE)
require (rlist)  # é installer si nécessaire 
require(keras)
install_keras()
install.packages("caret")
library(caret)
install.packages("tidyverse")
library(tidyverse)
install.packages("e1071")
library("e1071")
install.packages("kernlab")
library("kernlab")
install.packages("ggplot2")
library("ggplot2")

# 1. Le jeu de donnees MNIST # 

# mnist <- dataset_mnist() / cette commande n'a pas fonctionné
# on télécharge le jeu r.data
mnist <-list.load("\\\\filer-eleves.domensai.ecole\\id0551\\Apprentissage\\mnist.rdata")

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# visualize the digits
par(mfcol=c(6,6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in 1:36) {
  im <- x_train[idx,,]
  im <- t(apply(im, 2, rev))
  image(1:28, 1:28, im, col=gray((0:255)/255),
        xaxt='n', main=paste(y_train[idx]))
}
# Ca fonctionne, on visualise des images représentant des chiffres

# 2. Préparation des donnees #  .............................................. 3/3

# 2.1.1 On va aplatir les données pour les transformer en vecteur (de 784 pixel contre une image de 28*28)
?array_reshape
x_train2 <- array_reshape(x_train, c(nrow(x_train), 784))
colnames(x_train2) <- 1:784
x_test2 <- array_reshape(x_test, c(nrow(x_test), 784))
colnames(x_test2) <- 1:784
# 2.1.2. On cherche a supprimer les pixels nuls 
#?nearZeroVar
pixel_nul_train <- nearZeroVar(x_train2, uniqueCut = 0) # 67 pixels nuls
pixel_nul_test <- nearZeroVar(x_test2, uniqueCut = 0) # 116 pixels nuls
pixel_nul_global <- c(pixel_nul_test, pixel_nul_train) # ................................ 0.5 + 0.5 
# Il y a des pixels nuls identiques aux deux échantillons : 
#View(pixel_nul_global) .................
pixel_nul_global <- unique(pixel_nul_global)
# On choisi retire les pixels des échantillons :
x_train2 <- x_train2[, -pixel_nul_global]
x_test2 <- x_test2[, -pixel_nul_global]

dim(x_train2) # 60000*666
dim(x_test2) # 10000*666
# 2.2. La dimension (666 variables) étant encore importante on va essayer de la réduire 
# Il est demandé de maintenir la méme dimension entre l'échantillon test et entrainement
?preProcess
# La fonction preprocess permet de retariter les données avant apprentissage et fait partie du package caret 
# En particulier, si l'on indique dans l'option méthode "PCA" elle réalise une ACP 
# Elle la propiété de maintenir l'échelle des échantillons
# Dans notre cas, on peut eréaliser une ACP en supossant que chaque pixel représente une variable quati
# et que chaque ligne face référence ? un "invdividu

preProc  <- preProcess(x_train2, method = "pca")  # ...................................... 2/2
x_train2 <- predict(preProc, x_train2)
x_test2 <- predict(preProc, x_test2)
preProc
# Created from 60000 samples and 666 variables
# Pre-processing:
# - centered (666)
# - ignored (0)
# - principal component signal extraction (666)
# - scaled (666)
# PCA needed 303 components to capture 95 percent of the variance

dim(x_train2) # 60000*303
dim(x_test2) #10000*303
# On a bien respecté la contrainte de conservation des dimensions
# Automatique avec la fonction  preProcess (différent si on avait utilisé le package factominer)

# 3. Apprentissage par SVM linéaire # .............................. 1/2.5  pas de résultats 

set.seed(32123) # on fixe l'aléas
X_train3 <- cbind(x_train2, y_train)
X_test3 <- cbind(x_test2, y_test)

# Validation croisée ? 5 fold répétée 2 fois
control=trainControl(method="repeatedcv", number=5, repeats=2)


# Paramétre C
grid= expand.grid(C=10**(-3:0)) #c borne somme des erreurs (par rapport ? la distance de la droite séparatrice - hyperplan)
# En théroie, la métric accuracy permet d'obtenir le taux de classement (pour var réponse binaire)
# pour les régressions, faut prendre les moindres carrés. 
svm_linear <- svm.caret=train(y_train~.,data=X_train3, method="svmLinear", trControl=control, tuneGrid= grid)
# Impossible d'avoir le résultat, je l'ai laissée tournée plus de 2h sans solution
# La partie matrice de confusion n'a donc pas été testée
svm_linear
# On teste l'erreur de classement sur l'autre échantillon :
test_pred <- predict(svm_Linear, newdata = x_test)
confusionMatrix(test_pred, y_test)
# Le taux de bon classement du jeux de données test est de 

# 4. Apprentissage par réseaux de neurones artificiels # 
#Explication TP

# 5. Apprentissage par réseaux de neurones artificiels # .............................................4.5
# On repart des données initiales
# reshape
x_train3 <- array_reshape(x_train, c(nrow(x_train), 784))
x_test3 <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train3 <- x_train3 / 255
x_test3 <- x_test3 / 255

# 5.a 

model_RNA <- keras_model_sequential() 
# couche cachée comportant 784 LTU avec la fonction d'activation "relu".............................0.5 
layer_dense(model_RNA, units = 784, input_shape = c(784), activation = "relu")
# 10 LTU en sortie
layer_dense(model_RNA, units = 10, activation = "softmax") # 10 LTU en sortie
# on a besoin de 10 LTU sur la couche de sortie (de 0 ? 9) 
# La fonction d'activation est la fonction d'activation "softmax", qui transforme un vecteur de K nombres r?els en K probabilitées
print(summary(model_RNA)) # ........................................................................0.5

# 5.b ................................ 3.5/4.5 
# La fonction de perte est "entropie croisée" elle correspond ? la log vraissemblance 
# Comme on a plusieurs catégorie on prend la version catégorical et pas binary
# j'ai utilisé la version sparse_ car l'autre version généré des erreurs, celle ci permet de mieux traité les y_
# choix de la métrique "accuracy" dans le cas d'une classification - taux de succés 
# algorithme d'optimisation adam (Adaptative Moments) 
compile(model_RNA,loss = "sparse_categorical_crossentropy", optimizer = optimizer_adam(), metrics = 'accuracy')
# estimation des paramétres:epochs = nombre d'itération , batch_size = nombre d'observation
history3 <- fit(model_RNA, x_train3, y_train, epochs = 10, batch_size = 2000, validation_data = list(x_test3, y_test))
# Au vu des graphiques qui s'affiche la meilleure performance est obtenue sur le 10éme essai
# fonction d'évaluation de Keras
res <- evaluate(model_RNA, x = x_test3, y = y_test)
# affichage
print(res)
# 78,4 % de réussite 

# 6.Apprentissage par réseaux de neurones convolutifs #  ......................................................... 5
# Explication TP 

# 7. Implémentation d'un réseau de neurones convolutifs #  ......................... 9*0.25 =  2.25 + 0.25
# On essayera d'expliquer le réle de chaque couche au fur et à mesure
model_RNC <- keras_model_sequential()  
# 7.1 couche de convolution : permet de traiter l'information, un volume d'entree, pour fournir un volume de sortie, ici des images .....0/1
layer_conv_2d(model_RNC, filters = 30, kernel_size = c(5,5), activation = "relu", input_shape = c(28, 28, 1)) 
# 7.2 couche max-pooling 2d : permet de sous échantillonner l'image a traiter et donc de réduire le temps de calcul......... 0/1
layer_max_pooling_2d(model_RNC, pool_size = c(2,2)) 
# 7.3 couche de convulution, idem 7.1
layer_conv_2d(model_RNC, filters = 15, kernel_size = c(3,3), activation = "relu")
# 7.4 couche max-pooling 2d, idem 7.2
layer_max_pooling_2d(model_RNC, pool_size = c(2,2)) 
# 7.5 couche dropout : permet d'accéler l'apprentissage/ et d'éviter le surapprentissage en désactivant de faéon aléatoire certaine sortie de neurone (simule différend modéle)
# ...............................................................................................................1/1 
layer_dropout(model_RNC, rate = 0.3) 
# 7.6 couche d'applatissement: permet de mettre "a plat" les données, dans un seul vecteur qui permettra de connecter toutes les valeurs 
# ................................................................................................................1/1
layer_flatten(model_RNC)
# 7.7 couche ReLU (Rectified linear unit) : permet d'améliorer l'éfficacité du traitement / fait référence é la fonction d'activation dont elle permet d'augmenter les propriétés
# .................................................................................... 0/1
layer_dense(model_RNC, units = 128, activation = "relu")
# 7.8 couche ReLU idem 7.7
layer_dense( model_RNC, units = 50, activation = "relu")
# 7.9 couche softmax : comme pour la partie 5. elle transforme un vecteur de K nombres réels en K probabilitées.................0.5/0.5
layer_dense(model_RNC, units = 10, activation = "softmax")
print(summary(model_RNC))

# Le code ci-dessous est le méme que celui de la partie 5.
compile(model_RNC,loss = "sparse_categorical_crossentropy", optimizer = optimizer_adam(), metrics = 'accuracy')

# estimation des paramétres:epochs = nombre d'itération , batch_size = nombre d'observation
history4 <- fit(model_RNC, x_train, y_train, epochs = 10, batch_size = 2000, validation_data = list(x_test, y_test))
# Ca ne fonctionne pas, l'erreur suivante s'affiche : 
# Error in py_call_impl(callable, dots$args, dots$keywords) : 
# ValueError: Error when checking input: expected conv2d_4_input to have 4 dimensions, but got array with shape (60000, 28, 28) 
# Je n'ai pas réussi é résoudre le probléme

# fonction d'évaluation de Keras
res4 <- evaluate(model_RNC, x = x_test, y = y_test)
# affichage
print(res4)
