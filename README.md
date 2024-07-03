# Machine Learning Challange
This project was undertaken as part of the Machine Learning course conducted for the partial fulfilment of the MSc Data Science & Society at Tilburg University. It was a group project aimed at performing a binary classification task using three different machine learning models. The goal was to split the dataset, preprocess the data, train the models, optimize their parameters, and compare their performances.

## Splitting the Data
The data was split into a training, validation, and a test set. Given that 20% of the data had to be reserved for testing, the training and validation sets were assigned the remaining 60% and 20% of the data respectively. A smaller training set would yield greater variance in the training, while a smaller validation and test set would yield greater variance of the model's performance. In order to strike a balance between these premises, the majority of the dataset was reserved for training, while, mimicking the test set, the remaining 20% was reserved for validation. Within the workflow, the splitting was conducted prior to data preprocessing and other feature engineering tasks in order to avoid data leakage. The parameters obtained in the training set were used to normalize and oversample all sets. The split was conducted with the `train_test_split` function provided by the `sklearn.model_selection` library. The random state argument was used to ensure the reproducibility of the split.

## Feature Transformations and Selection
The following transformations were made:

- **Encoding categorical variables:** The variable 'Customer_Type' was one-hot encoded using the `OneHotEncoder` function provided by the `sklearn.preprocessing` library. The category with the highest value count, 'Returning_Customer', was used as reference.
- **Outlier detection and imputation:** A very extreme value was detected in the training set ('ProductPage_Time' with a value of 63973.52, given a mean of 1180.33). This datapoint was deleted and later the mean of the column was imputed in its place.
- **Scaling:** Many of the features in the dataset have a small range (such as 0-1), however, some of the features have a comparatively large range (in the thousands). Given the skewness of the features, normalization was preferred over standardization. The three sets of data were set to have 0 mean and unit variance in order to prevent undue influence of variables with larger scales and to aid faster convergence. The `MinMaxScaler` method from the `sklearn.preprocessing` library was employed.
- **Handling imbalanced classes:** The target variable 'Transaction' was oversampled in order to avoid bias towards the majority class. Oversampling was preferred over undersampling given the modest count of instances.

Since the meaning of the features could only be inferred from the column labels, an agnostic feature selection process was employed for the three algorithms, aiming to identify the most informative features. The permutation importance of the features was computed in order to obtain an importance score, computed as the difference between the baseline model performance and the performance of the permuted model. Features were filtered to only include those with an importance score greater than zero. The obtained variables include: 'SystemF1', 'SystemF2', 'SystemF3', 'SystemF4', 'SystemF5', 'Account_Page', 'Account_Page_Time', 'Info_Page_Time', 'ProductPage', 'ProductPage_Time', 'Month', 'GoogleAnalytics_PV', 'Customer_Type_Returning_Customer'.

## Learning Models and Algorithms
### Baseline Model: Logistic Regression
Logistic regression was chosen as the baseline model as it is a commonly used algorithm for binary classification tasks, due to its simplicity, interpretability, and ability to identify linear decision boundaries.

The model employs a logit function to map the input to a 0-1 probability, thus directly modeling the probability of class membership and allowing for straightforward probability estimation.

#### Hyperparameters
Grid search was performed on the logistic regression model trained on the selected features. The model was fitted on the training data with each combination of hyperparameters, to then establish the best set of hyperparameters by considering accuracy as a scoring metric. The two hyperparameters considered are the regularization and penalty parameters, with the following values respectively: `C`: [0.1, 1, 10], `penalty`: ['l1', 'l2']. The accuracy score determined the final choice: `C`: 1, `penalty`: 'l2'.

### Support Vector Machine (SVM)
SVM was chosen due to its effectiveness in high-dimensional spaces, ability to handle non-linear decision boundaries using the kernel trick, robustness to outliers, suitability for imbalanced data with the use of SMOTE oversampling, and the flexibility to optimize performance through hyperparameter tuning.

#### Hyperparameters
The following hyperparameters were tuned for the SVM model using `GridSearchCV`:
1. **C:** The C parameter controls the trade-off between maximizing the margin and minimizing the training error. In the code, the values [0.1, 1, 10] are explored for C.
2. **Kernel:** The kernel parameter determines the type of kernel function used to transform the input data into a higher-dimensional space. The two kernel options explored in the code are 'linear' and 'rbf' (radial basis function).
3. **Gamma:** The gamma parameter defines the influence of a single training example. The values 'scale' and 'auto' are explored for gamma in the code.

The `.best_params_` attribute from `GridSearchCV` was used to determine the best performing combination of hyperparameters: `C`: 10, `gamma`: 'scale', `kernel`: 'rbf'.

### Decision Tree (DTs)
The DT model is used for both regression and classification tasks. In classification tasks, DTs are effective due to their simplicity and interpretability (Song & Ying, 2015). They don't make any distribution assumptions (non-parametric model), as opposed to SVM. Their transparency is also higher because the branches and leaves show the decision making process clearly. DTs are also more robust to outliers compared to logistic regression (Song & Ying, 2015).

#### Hyperparameters
A grid search was performed to establish the best hyperparameters by looking at accuracy scores. DTs are less sensitive to hyperparameter tuning than SVM, however with hyperparameter tuning the model performs better (Mantovani et al., 2016). Parameters include Gini impurity (probability of misclassification) and Entropy (degree of uncertainty) (Breiman, 1996). Another parameter considered was the depth of the tree, where a deeper tree shows more complex relationships but could potentially overfit. Considered depths include: None, 5, 10 and 15. The final parameters used were Entropy and a maximum depth of 15.

## Discussion & Results

|                        | Logistic Regression | SVM   | Decision Tree |
|------------------------|---------------------|-------|---------------|
| **Accuracy Training**  | 0.8226              | 0.8851| 1.0000        |
| **Accuracy Validation**| 0.8693              | 0.8742| 0.8513        |
| **Accuracy Test**      | 0.8640              | 0.8615| 0.8410        |
| **Precision**          | 0.5676              | 0.5789| 0.5274        |
| **Recall**             | 0.7195              | 0.7429| 0.5506        |
| **F1 Score**           | 0.6346              | 0.6507| 0.5388        |

## Comparing the Models
Based on these results, the SVM model performs the best overall. It achieves high accuracy scores on the training, validation, and test sets, indicating strong predictive capability. The SVM model also demonstrates the highest precision, recall, and F1 score, indicating better performance in correctly classifying both positive and negative instances compared to the other models. While the logistic regression model also shows good performance, the SVM model outperforms it in terms of accuracy, precision, recall, and F1 score. The decision tree model performs the poorest among the three models, with lower accuracy and lower precision, recall, and F1 score.

## Transparency and Reproducibility
The code provided demonstrates some transparency and reproducibility. The code is well-documented with comments explaining the purpose of each step and the libraries being used. This makes it easier for others to understand and reproduce the analysis. Additionally, the code includes the necessary import statements and specifies the data preprocessing steps, model training, hyperparameter tuning, and evaluation metrics.

## Strengths and Weaknesses of the Method Implementation and Current Analysis
### Strengths:
1. It includes visualizations, such as boxplots and confusion matrices, to provide insights into the data and model performance.
2. The code handles class imbalance using the SMOTE oversampling technique.
3. It uses different models (Logistic Regression, SVM, DT) and compares their performance.
4. The code evaluates the models using multiple metrics such as accuracy, confusion matrix, precision, recall, and F1 score, providing a comprehensive assessment of model performance.

### Weaknesses:
1. The code does not consider potential overfitting of the data, particularly within the DT model.
2. Even though the three models correctly classify most instances, with SVM displaying the greatest accuracy, the obtained F1 scores are not particularly high. Other algorithms such as Neural Networks and Random Forests could be explored in order to obtain improved F1 scores.

## Conclusion
To conclude, while the code covers the most important steps of a machine learning pipeline, it would benefit from consideration of potential overfitting, and further analysis by employing more complex algorithms. Additionally, providing more information about the dataset and ensuring complete reproducibility would enhance the transparency and usability of the code.
