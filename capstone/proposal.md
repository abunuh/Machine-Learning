# Machine Learning Engineer Nanodegree
## Capstone Proposal
Ebrahim Jakoet

August 10th, 2019

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

About 12% of women living in America will develop some form of invasive breast cancer.  It is estimated that 41,760 of women in the US will die from breast cancer in 2019.  Breast cancer remains among the most commonly diagnosed types of cancer in older women especially, and a leading cause of death compared to most other types of cancer [#1].  This is based on the latest available information on [BreastCancer.org](https://www.breastcancer.org/symptoms/understand_bc/statistics).

In this project we will use the Breast Cancer Wisconsin (Diagnostic) Data Set to build a model that can predict the presence of a malignant cancer based on the given features in the dataset.  We will also determine which features are more significant in the detection of malignant breast cancer.  The dataset is derived from digitized images of a fine needle aspirate (FNA) biopsy of breast mass samples. The features describe characteristics of the cell nuclei present in the image [^2].  Early detection of breast cancer can be life-saving and Machine Learning is increasingly being relied upon as more reliable in early detection methods.  It is this that draws me to the subject.

![Digitized image of a fine needle aspirate (FNA) of a breast mass](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwilkLyjlfbjAhXuct8KHSk0AmYQjRx6BAgBEAQ&url=%2Furl%3Fsa%3Di%26rct%3Dj%26q%3D%26esrc%3Ds%26source%3Dimages%26cd%3D%26ved%3D%26url%3D%252Furl%253Fsa%253Di%2526rct%253Dj%2526q%253D%2526esrc%253Ds%2526source%253Dimages%2526cd%253D%2526ved%253D2ahUKEwjUzeLskfbjAhUJR6wKHW0ADUMQjRx6BAgBEAQ%2526url%253Dhttps%25253A%25252F%25252Fwww.kaggle.com%25252Fuciml%25252Fbreast-cancer-wisconsin-data%2526psig%253DAOvVaw1_jUt7e07JHOPEi-gxI7DN%2526ust%253D1565451928059637%26psig%3DAOvVaw1_jUt7e07JHOPEi-gxI7DN%26ust%3D1565451928059637&psig=AOvVaw1_jUt7e07JHOPEi-gxI7DN&ust=1565451928059637)



### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

There are 30 feature measurements contained in the dataset with each sample labeled as either (M) Malignant or (B) Benign.  The goal is to determine which features are most significant and to build and compare an ensamble model and a DNN model that will be able to predict the malignancy of the FNA sample based on the given features.  The model results will be compared as well. The procedure will be as follows:
1. Extract the raw data from the compressed zip file.
2. Explore the data and ensure that it is cleaned before building any prediction models.
3. Do some basic feature selection and consider removing highly correlated features.
4. Split the data into training and testing sets.
4. Train an ensable model like AdaBoost and a DNN on the training set of the data.
5. Review the results for accuracy of the models.  Compare the ensamble model to the DNN.
6. Review the results for feature importance and compare to the feature selection used earlier.



### Datasets and Inputs
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

The data set was aquired from the Kaggle website [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) project.  It is also available from the [UCI Website Archive](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).  The data consists of a single file of 32 columns and 569 sample instances.  Creators of the data set are listed below:

1. Dr. William H. Wolberg, General Surgery Dept.
University of Wisconsin, Clinical Sciences Center
Madison, WI 53792
wolberg '@' eagle.surgery.wisc.edu

2. W. Nick Street, Computer Sciences Dept.
University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
street '@' cs.wisc.edu 608-262-6619

3. Olvi L. Mangasarian, Computer Sciences Dept.
University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
olvi '@' cs.wisc.edu

The data consists of an ID column, a labeled target column that shows (M) for malignant or (B) for Beinign and the following 10 features.

1. radius (mean of distances from center to points on the perimeter) 
2. texture (standard deviation of gray-scale values) 
3. perimeter 
4. area 
5. smoothness (local variation in radius lengths) 
6. compactness (perimeter^2 / area - 1.0) 
7. concavity (severity of concave portions of the contour) 
8. concave points (number of concave portions of the contour) 
9. symmetry 
10. fractal dimension ("coastline approximation" - 1)

Each of these 10 features have a mean, standard error and worst or largest (mean of the three largest values) value which makes the total feature set 3 x 10 = 30 features in total.  Feature values have been recorded with 4 significant digits.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

The solution involves the building of 2 competing models.  The first model will be an AdaBoost ensamble model and will be used as the benchmark model.  The second model will be a Deep Neural Network (DNN) built with the Keras modules for neural network architecture.  Each model will produce an accuracy and recall score comparing the result of training and testing data using the following formulae:

`Accuracy = True Positives + True Negatives / Total number of Samples`

`Recall = True Positives / (False Negatives + True Positives)`

Since this is a test for the presence or absence of a malignant breast mass, the Recall evaluation metrics will also be relevant. 

We will also compare the processing time for each learning algorithm.  Ideally we are looking for a solution with high accuracy, but the lower processing time may be relevant if processing is a limiting factor for the actual deployment.

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

We will use the AdaBoost model as a benchmark model since we are comparing Adaboost to a DNN.  There are many hyper-parameters that can be set for the AdaBoost model.  For our model we will use the following setup, but these hyper-parameters will likely be changed to get the highest evaluation metrics possible.

1. base_estimator: DecisionTreeClassifier(max_depth=1)
2. n_estimators: 100
3. learning_rate: 1.0
4. algorithm: SAMME.R
5. random_state: 19




### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

To compare the AdaBoost and DNN models, we will be using an accuracy score, recall and processing time.  Accuracy will give us a measure of how well the model is able to predict the correct outcome.  Recall is important because we want to be able to minimize the False Negatives since this is a test for the presence of cancer in a breast mass.  Processing time is only relevant if the deployment of the solution has limited resources.  It is always good to see how the 2 models compare in time.  The The formulae for the accuracy and recall metrics are shown below.


`Accuracy = True Positives + True Negatives / Total number of Samples`

`Recall = True Positives / (False Negatives + True Positives)`

For processing time we will use the python time modules to calculate time difference between the training start and stop of each of the 2 models.


### Project Design
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

### References


- [^1]: `https://www.breastcancer.org/symptoms/understand_bc/statistics`.

- [^2]: W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993. 
-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
