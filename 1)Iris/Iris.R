
str(Iris)

Iris$new_class_species <- as.character(iris$Species)
Iris$new_class_species <- NULL
Iris$Species <- gsub("%","",Iris$Species)
Iris <- na.omit(Iris)

summary(iris)
plot(iris)

par(mfrow=c(1,2))
plot(iris$Petal.Length)
boxplot(iris$Petal.Length~iris$Species)

par(mfrow=c(2,2))
for(i in 1:4) boxplot(iris[,i]~Species,data=iris,main=names(iris)[i])

  