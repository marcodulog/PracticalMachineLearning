Prediction Assignment
---------------------

The following report is a Coursera assignment to create a model that will accurate predict the "classe" of exercise provided. We are asked to provide a report on:

1.  How the model was built?

2.  How we used cross validation?

3.  What the expected out of sample error was?

4.  Justify the choices made.

5.  Run the prediction against 20 different test cases.

All the data provided was provided by the coursera assignment but was sourced from the following:

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6\_6.

### Require libraries

The following libraries are required for this evaluation

``` r
require(caret)
require(ggplot2)
require(rpart)
require(randomForest)
require(dplyr)
```

### Importing data

Load the training and testing data provided. The data is pretty messy and needs to be cleaned up. The first seven columns don't really work well with prediction and can be removed. The remaining variables have varying degrees of NA values that can be removed as well.

``` r
# import the training and testing dataset
training<-read.csv("C:/Files/Coursera/R/PracticalMachineLearning/pml-training.csv",na.strings=c("NA", "#DIV/0!", ""))
testing<-read.csv("C:/Files/Coursera/R/PracticalMachineLearning/pml-testing.csv",na.strings=c("NA", "#DIV/0!", ""))

#prepare the original training dataset
removeNA = sapply(1:dim(training)[2],function(x)sum(is.na(training[,x])))
removeNACols = which(removeNA>0)
training = training[,-removeNACols]
training = training[,-c(1:7)]
training$classe = factor(training$classe)


#prepare the original testing dataset
removeNA = sapply(1:dim(testing)[2],function(x)sum(is.na(testing[,x])))
removeNACols = which(removeNA>0)
testing=testing[, apply(testing, 2, function(x) !any(is.na(x)))] 
testing=testing[,-c(1:7)]
```

### Create the cross validation dataset

Partition the training dataset into at 60% training and 40% testing to see what type of errors we will receive and choose the more accurate model before applying it to the actual testing dataset.

``` r
# split the cleanTrainData into 75% and 25%
inTrain <- createDataPartition(y=training$classe, p=0.60, list=FALSE)
train<-training[inTrain,]
test<-training[-inTrain,]
```

### Create the models

First we will evaulate the accuracy of the two data models using cross validation

1.  Decision Tree

2.  Random Forrest

``` r
# Decision Trees
modTree <- train(classe ~ .,method='rpart',data=train)
predTree <- predict(modTree,test)

# Random Forrest
modForrest <-randomForest(classe~., data=train, method='class')
predForrest <- predict(modForrest,test )
```

### Evaluate the models

Determine the accuracy of the models by using the predict function on the training (test - which was not part that was used to create the model) dataset against both the Decision Tree model (modTree) and the Random Forrest Model (modForrest).

#### Accuracy of the Decision Tree Model

We are not expecting the greatest results with the Decision Trees and as the accurancy shows, we have under 50% accuracy rate

``` r
confusionMatrix(predTree, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2026  648  631  564  209
    ##          B   44  522   41  235  171
    ##          C  157  348  696  487  408
    ##          D    0    0    0    0    0
    ##          E    5    0    0    0  654
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4968          
    ##                  95% CI : (0.4857, 0.5079)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3424          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9077  0.34387  0.50877   0.0000  0.45354
    ## Specificity            0.6345  0.92241  0.78388   1.0000  0.99922
    ## Pos Pred Value         0.4968  0.51530  0.33206      NaN  0.99241
    ## Neg Pred Value         0.9453  0.85424  0.88313   0.8361  0.89036
    ## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
    ## Detection Rate         0.2582  0.06653  0.08871   0.0000  0.08335
    ## Detection Prevalence   0.5198  0.12911  0.26714   0.0000  0.08399
    ## Balanced Accuracy      0.7711  0.63314  0.64633   0.5000  0.72638

#### Accuracy of the Random Forrest Model

The accuracy of the Random Forrest model is fairly accurate to about 99.26%. We will choose this as our prediction model.

``` r
confusionMatrix(predForrest, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2230    8    0    0    0
    ##          B    2 1506   10    0    0
    ##          C    0    4 1358   15    0
    ##          D    0    0    0 1269    3
    ##          E    0    0    0    2 1439
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9944          
    ##                  95% CI : (0.9925, 0.9959)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9929          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9991   0.9921   0.9927   0.9868   0.9979
    ## Specificity            0.9986   0.9981   0.9971   0.9995   0.9997
    ## Pos Pred Value         0.9964   0.9921   0.9862   0.9976   0.9986
    ## Neg Pred Value         0.9996   0.9981   0.9985   0.9974   0.9995
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2842   0.1919   0.1731   0.1617   0.1834
    ## Detection Prevalence   0.2852   0.1935   0.1755   0.1621   0.1837
    ## Balanced Accuracy      0.9988   0.9951   0.9949   0.9932   0.9988

We can see that the results for the Random Forrest are very high with a 99% accuracy rate

### Prediction of the testing dataset

``` r
predSubmission <- listing<-predict(modForrest,testing,type='class')
listing
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

\`\`\`
