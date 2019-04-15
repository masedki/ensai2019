#install.packages("caret")
#install.packages("MLmetrics")
#install.packages("e1071")
#install.packages("keras")
library(e1071)
library(MLmetrics)
library(caret)
library(factoextra)
require(keras)

# 1-........................................................................................ 2.25
xmnist <- dataset_mnist()
x_train <- mnist$train$x
# On change les noms de niveaux du facteur car pour l'entraînement du svm linéaire, ils doivent correspondre
# à des noms de variables valides (ne pas commencer par un chiffre, notamment)
y_train <- factor(mnist$train$y,labels = c("X0","X1","X2","X3","X4","X5","X6","X7","X8","X9"))


x_test <- mnist$test$x
y_test <- factor(mnist$test$y,labels = c("X0","X1","X2","X3","X4","X5","X6","X7","X8","X9"))

# visualize the digits
par(mfcol=c(6,6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in 1:36) {
  im <- x_train[idx,,]
  im <- t(apply(im, 2, rev))
  image(1:28, 1:28, im, col=gray((0:255)/255),
        xaxt='n', main=paste(y_train[idx]))
}
# 2-

# applatissement des données
x_train <- array_reshape(x_train, c(nrow(x_train), 784))#......................................................... 0.25
x_test <- array_reshape(x_test, c(nrow(x_test), 784))#............................................................ 0.25

# Affichage de la dimension avant la réduction de dimension
print(ncol(x_train))

# On repère les indices des pixels qui ne portent aucun information dans l'ensemble d'apprentissage
indicesNuls <- nearZeroVar(x_train)#...............................................................................0.25
# Et on les ignore dans les ensembles d'apprentissage et de test
x_train <- x_train[,-indicesNuls]
x_test <- x_test[,-indicesNuls]

# On fait une acp a la main pour savoir combien de composantes factorielles garder
acp <- prcomp(data.frame(x_train))
fviz_eig(acp)
plot(summary(acp)$importance[3,])

# finalement on garde la valeur par défaut : conserver 95% de variance projetée
# les données sont également centrées et réduites
preproc <- preProcess(data.frame(x_train), thresh = 0.95, method = "pca")#......................................1.5/2 on prend tous les axes
x_train_pred <- predict(preproc, data.frame(x_train))
x_test_pred <- predict(preproc, data.frame(x_test))

# On regroupe les données dans des data frames
train <- data.frame(x_train_pred,y_train)
test <- data.frame(x_test_pred,y_test)

# Il reste 90 variables (composantes factorielles)
print(ncol(x_train_pred))

# 3. Apprenstissage par SVM linéraire ................................................................ 2.5

# Setup for cross validation
ctrl <- trainControl(method="repeatedcv",
                     number = 5,
                     repeats=2,
                     summaryFunction=multiClassSummary,	
                     classProbs=TRUE)

tGrid <- data.frame(C = 10**(-3:0))

modelFit <- train(y_train ~ . , data = train,method="svmLinear",metric="Accuracy",trControl=ctrl,tuneGrid=tGrid)
# Affichage de la précision estimée par cross validation
trellis.par.set(caretTheme())
plot(modelFit, scales = list(x = list(log = 10)))
dev.off()
predictTest <- predict(object=modelFit,newdata = x_test_pred)

# 58,75 % d'images bien classées dans l'ensemble de test
print(mean(predictTest == y_test))
