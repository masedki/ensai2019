require(keras)


mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

#####################################################
# 5. Réseau de neurone à une couche cachée          #............................. 1 +  4 = 5
#####################################################

# Pour une tâche de classification, la dernière couche d'un réseau de neurones
# doit comporter un neurone par modalité de la variable expliquée, donc ici 10 ............... 0.5 + 0.5

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 784, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 10, activation = 'softmax') # Dans le cas d'une classification, il est recommandé d'utiliser une fonction d'activation softmax pour la couche de sortie

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 200, 
  validation_split = 0.2
)

plot(history)
# Le meilleur taux estimé par validation croisée est de 0.9795833 pour la 10ème période d'entraînement
print(history$metrics)

model %>% evaluate(x_test, y_test) # 98,03 % de bonnes prédictions sur l'ensemble de test
#model %>% predict_classes(x_test)

#######################################################
# 7. Réseau de neurones convolutifs........................................................................................ 7 
#######################################################
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)#............................................................... 0.25 

model <- keras_model_sequential() #.................................................................. 9 * 0.25 = 2.25
model %>% 
  # La première couche de convolution permet de construire des "features" à partir de sous parties des images
  # le nombre de connexion étant beaucoup plus restreint que dans un réseau classique, le temps d'entraînement est
  # beaucoup plus court. .......................................................................................................... 1/1
  layer_conv_2d(filters = 30, kernel_size = c(5,5), activation = 'relu', 
                input_shape = c(28,28,1)) %>% # Les images sont de dimension 28*28 sur un seul canal étant donné au'on utilise des niveaux de gris
  # la couche de pooling permet de réduire la dimension (conserve les caractéristiques les plus saillantes de l'image
  # en gardant la valeur maximale pour noyau (groupe de pixel)). Il s'agit d'une forme de compression avec perte destinée
  # à réduire l'importante dimension en sortie de la couche de convolution......................................................... 1/1
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  # On réapplique une couche de convolution et une couche de pooling, l'idée est la même que précédemment
  layer_conv_2d(filters = 15, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  # Le dopout permet d'éviter le suraprentissage en supprimant les connexions les moins "significatives" ...........................FAUX
  layer_dropout(rate = 0.3) %>% 
  # On réapplatit la matrice de pixel afin de passer les caractéristiques obtenues par les couches de convolution .................1/1
  layer_flatten() %>% 
  # Un réseau de neurone classique avec une couche cachée de 50 neurones prend en entrée les abstractions de l'image ainsi obtenues
  # et se charge de faire le travail de classification en lui-même. ............................................................ 1.5/1.5
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

# Le 1 correspond au fait qu'il n'y a qu'un seul canal car l'image est codée en niveaux de gris
x_train_2 <- array(x_train, dim = c(60000,28, 28, 1))  

history <- model %>% fit(
  x_train_2, y_train, 
  epochs = 10, batch_size = 200, 
  validation_split = 0.2
)

plot(history)
# Le meilleur taux estimé par validation croisée est de 0.9895000 pour la 9ème période d'entraînement
print(history$metrics)

model %>% evaluate(x_test, y_test)
#model %>% predict_classes(x_test)

