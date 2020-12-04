# Notes

## Feature Selection

* We can remove words associated with only 1 particular blurb
* TfIdfVectorizer paramaters, `max_df`, `min_df`

## Tweakable Hyperparamaters

* TF-IDF vs TF
  * IDF weights down words that appear in multiple documents
    * We may not want to do that because we are considering frequency across genre

## Considerations

* Does PCA work with non-linear relationships?
* If logistic classifier performs badly it may indicate a non-linear relationship, in which case try SVM with kernel trick or Neural Network
  * For NN, if running takes long, research if we can use Collab given project structure or Google Cloud or NYU HPC
* to do cross validation, combine training, dev, and test into one df

### Genres

* Pick more sub-genres for promotion and remove base genres (i.e. d0)
* Run only on d1s or others

### Cloud Computing

https://cloud.google.com/tpu/pricing

### Analysis

* Confusion matrix plot
* accuracy, mse, f1, recall
  * per class
* Comparisons between models
  * n_components = 100 vs 200
  * max_df lowered pov when max_df = 1 / num_classes
  * no min_df lowered pov
  * using IDF gave better results
* due to slow iteration times with svm we decided to use compliment niave bayes during feature selection
  * cmb 


## TODO

* keep only min samples over all classes for each class
  * i.e. business is smallest class with 650 items so make all classes 650 sized
* remove children's books