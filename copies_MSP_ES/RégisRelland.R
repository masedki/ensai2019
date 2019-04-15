
# ------------------ Projet apprentissage Mnist -----------------------------------
#Regis Relland


#..................................................................................
# 1.Le jeu de donn?es MNIST
#..................................................................................

#chargement initial des donn?es
rm(list=ls())
setwd("P:/Apprentissage")
install.packages("keras", dep=TRUE)
require(keras)
# mnist <- dataset_mnist()

#sauvegarde et chargement des donn?es
require (rlist)
list.save(mnist, file='mnist.rdata')
mnist <-list.load('mnist.rdata')

#Jeux train, test
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



#..................................................................................
# 2.Pr?paration des donn?es                                               ................................................. 3
#..................................................................................


# on aplatit les carr?s 28x28 de chaque image en un vecteur 1x728
#c(nrow(x_train, 784)) -> (60000 ; 784) c'est la dimension qu'on souhaite
x_train_aplati <-array_reshape(x_train, c(nrow(x_train), 784)) #......................................................... 0.5
x_test_aplati <-array_reshape(x_test, c(nrow(x_test), 784))


#enlever les pixels nuls, les endroits presque toujours vides,
#donc avec une variance quasi nulle
library(caret)
# je remets momentanement ensemble les jeux de train et test dans data
data <-rbind(x_train_aplati, x_test_aplati) # ........................................................................... 0.5
nzv <- nearZeroVar(data)
data_nzv <- data[, -nzv]
#il reste 250 colonnes sur les 784


#reduire la dimension avec la fonction preprocess, pca fait une ACP qui conserve 95% variance ........................... 2
colnames(data_nzv) <- c(1:250)
param_preProcess <- preProcess(data_nzv, c("center", "scale", "pca"))
print(param_preProcess)
data_acp <- stats::predict(param_preProcess, data_nzv)
# avec l'acp on reduit ? 90 variables au lieu de 250

# je s?pare train et test comme c'etait auparavant 60000/10000
x_train_reduit <- data_acp[1:60000,]
x_test_reduit <- data_acp[60001:70000,]

rm(list=c('data', 'data_acp', 'data_nzv','mnist', 'x_test_aplati', 'x_test_nzv',
         'x_train_aplati','x_train_nzv', 'nzv'))



#..................................................................................
# 3. SVM lin?aire                                                                  ............................... 2.5 
#..................................................................................

install.packages("e1071")
require(e1071)


#recherche des meilleurs parametres de svm
control = caret::trainControl(method="repeatedcv", number=5, repeats=2)
# C= cost = le cout de violation des contraintes
grid=expand.grid(C=c(0.001, 0.01, 0.1, 1))
svm.caret <- caret::train( x_train_reduit,  factor(y_train),
                           trControl = control, tuneGrid=grid, method="svmLinear") 
svm.caret$bestTune
#c=0.1 (apres plusieurs heures !)


#svm lineaire avec c=0.1 sur le jeu train
outSvm <- svm(x_train_reduit,  factor(y_train), kernel="linear", cost=0.1)

#prediction sur echantillon test
predSvm <- stats::predict(outSvm, x_test_reduit)

#taux de mal class?s:   6.78%
mean(predSvm != y_test)



#..................................................................................
# 4 et 5. Apprentissage par r?seaux de neurones artificiels ....................... 4.5 + 0.5 + 0.5
#..................................................................................

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255


#structure de r?seau
model <- keras_model_sequential()

model %>%
  layer_dense(units = 784, input_shape = 784) %>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 10) %>%
  layer_activation(activation = 'softmax')


model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)


model %>% fit(x_train, y_train, epochs = 10, batch_size = 200)


#fonction d'?valuation de Keras
res <- model %>% evaluate(
    x = x_test,
    y = y_test
  )

#taux d'erreur: 1.87%
print(100*(1-res$acc))

  
  
  
#..................................................................................
# 6 et 7. Apprentissage par r?seaux de neurones convolutifs ............................. 2.5
#..................................................................................
  

mnist <-list.load('mnist.rdata')

#Jeux train, test
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# je laisse les images en 28x28 mais je fais les deux lignes suivantes
# pour avoir le bon format dans l'input_shape du model2
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255
 
  
  
model2 <- keras_model_sequential() %>%
    layer_conv_2d(filters=30, kernel_size = c(5,5), input_shape = c(28,28,1), activation = 'relu' ) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_conv_2d(filters=15, kernel_size = c(3,3),  activation = 'relu' ) %>%
    layer_max_pooling_2d(pool_size = c(2,2)) %>%
    layer_dropout(rate=0.3) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dense(units = 50, activation = 'relu') %>%
    layer_dense(units = 10, activation = 'softmax')
  
model2 %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = c('accuracy')
  )
  
  
model2 %>% fit(x_train, y_train, epochs = 10, batch_size = 200)
  
  
#fonction d'?valuation de Keras
res2 <- model2 %>% evaluate(
    x = x_test,
    y = y_test
  )

# taux d'erreur: 0.88%
 print(100*(1-res2$acc))
 