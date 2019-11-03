# Machine Learning Engineer Nanodegree

## Capstone Project: Facial Keypoints Detection
Daniel Vargas  
October 21, 2019


## I. Definition

### Project Overview

**Facial Keypoints `(facial landmarks)` detection** is an important and challenging problem in the field of **computer vision**, which involves detecting facial keypoints like centers and corners of `eyes`, `nose`, and `mouth`, etc. The problem is to predict the `(x, y)` real-valued coordinates in the space of image pixels of the facial keypoints for a given face image. 

Facial features vary greatly from one individual to another, and even for a single individual there is a large amount of variation due to `pose`, `size`, `position`, etc. The problem becomes even more challenging when the face images are taken under different `illumination conditions`, `viewing angles`, etc.

Solving this problem that can provide the building blocks for several applications, such as:

- tracking faces in images and video
- analysing facial expressions
- detecting dysmorphic facial signs for medical diagnosis
- biometrics / face recognition

In the past few years, advancements in facial keypoints detection have been made by implementing **`Deep Convolutional Neural Networks (DCNN)**`.

Relevant academic research on this domain can be found in

- [Facial Keypoints Detection](https://pdfs.semanticscholar.org/f9a3/072ee0a579475f5b5359e782ea32c2cbf24c.pdf)
- [Facial Key Points Detection using Deep Convolutional Neural Network - NaimishNet](https://arxiv.org/pdf/1710.00977.pdf).

I chose this specific challenge because I currently work in the medical diagnosis field. I expect this project to help me understand facial keypoints recognition in a deeper way.

### Datasets and Inputs

The data was acquired from the [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/overview) `Kaggle` competition. Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. There are 15 keypoints, which represent the following elements of the face:

|               |               |               |
| ------------- | ------------- | ------------- |
| `left_eye_center`  | `right_eye_center`  | `right_eye_center`  |
| `left_eye_inner_corner`  | `left_eye_outer_corner`  | `right_eye_inner_corner`  |
| `left_eyebrow_inner_end`  | `left_eyebrow_outer_end`  | `right_eyebrow_inner_end`  |
| `right_eyebrow_outer_end`  | `nose_tip`  | `mouth_left_corner`  |
| `mouth_right_corner`  | `mouth_center_top_lip`  | `mouth_center_bottom_lip`  |

- _Left and right here refers to the point of view of the subject_
- _In some examples, some of the target keypoint positions are misssing (encoded as missing entries in the csv, i.e., with nothing between two commas)_
- _The input image is given in the last field of the data files, and consists of a list of pixels (ordered by row), as integers in (0,255). The images are 96x96 pixels_

### Data files

- [`training.csv`](https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip): list of `7049 training images`. Each row contains the (x,y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.
- [`test.csv`](https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip): list of `1783 test images`. Each row contains ImageId and image data as row-ordered list of pixels

### Problem Statement

The objective of this project is to accurately predict the facial keypoints (facial landmarks) of a face image. My hypothesis is, that this prediction can be performed based on a training set containing accurate facial keypoints, through a regression approach.

A `Convolutional Neural Network` (`CNN`) will be applied to predict the facial keypoints. A `CNN` was chosen for this problem because:

- This is a computer vision problem that requires capturing features for prediction
- CNNs are very useful in capturing features in images
- The expected responses (coordinates) make this a regression problem

_A simple `linear regression` will be used as a baseline model for comparison._

### Metrics

The metric used to measure performance of the model i `Root Mean Squared Error` (`RMSE`):

![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5En%20%28x_i%20-%20y_i%29%5E%202%7D)

`RMSE` is very common and is a suitable general-purpose error metric in regression problems. Compared to the `Mean Absolute Error`, `RMSE` punishes large errors.

### Network Strategy

- Data augmentaion will be included if results are not satisfactory
- The network's artitecture is as follows:
    - Input layer
    - Convolution layers
    - Max Pooling layers
    - Batch Normalization layers
    - Fully Connected layers
    - Dropout layers
    - Prediction layer


## II. Analysis

### Data Exploration
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

Data Augmentation helps boost the performance of a deep learning model when the training data is limited by generating more training data. We have horizontally flipped [7] the images for which target information for all the 15 FKPs are available, and also swapped the target values according to Table 3. Then, we vertically stacked the new horizontally flipped data under the original train data to create the augmented train dataset.

Data Preprocessing

The image pixels are normalized to the range [0, 1] by dividing by 255.0, and the train targets are zero-centered to the range [-1, 1] by first dividing by 48.0, since the images are 96 x 96, and then subtracting 48.0.

Figure 1 shows that there are different number of non-missing target rows for different FKPs, so, we have created 15 NaimishNet models.


The dataset provided by Talking Data on Kaggle competition homepage includes approximately 200 million registered clicks over 4 days, split into training and testing sets. The training set contains more than 180 million rows of data, each has the timestamp of the click, number-encoded IP addresses, device numbered label code, device's operating system code, app code, channel code, whether the click resulted in a download or not, and time of download if applicable. The testing set contains about 18 million clicks with each click associated with an ID and other information excluding the download or not label and download time.

Files:

train.csv (7537.65 MB): the training data
test.csv (863.27 MB): the test data
train_sample.csv (4.08 MB): a number of randomly-selected rows from train.csv
Line counts:

train.csv: 184,903,890
test.csv: 18,790,469
train_sample.csv: 100,000
Labels:

On both sets:

ip (integer): IP address from which the click was registered, encoded for privacy
app (integer): the app whose advertisement was clicked on, encoded for privacy
device (integer): label of the model of the device on which the click was made
os (integer): label of device's operating system
channel (integer): the channel on which the app advertisement was put and clicked
click_time (yyyy-mm-dd hh-mm-ss): the time when the click was made
Training set:

attributed_time (yyyy-mm-dd hh-mm-ss): time of download if a download was made, None otherwise
is_attributed: whether this click resulted in a download or not; 1 - yes, 0 - no
Testing set:

click_id: used for identifying the click for grading purpose


![sample image](http://url/to/img.png)










### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_



The plot below shows how skewed the target concept is in the first 10 million rows of the training set:

![eda target distribution](assets/eda_target.png)
Clearly, this needs to be taken into consideration later when the classification model is chosen.

For each given feature, how many unique values does it have?

![eda unique values](assets/eda_uniques_per_feature.png)
While it makes sense for the number of unique devices and oses to be so low as there are only so many phone models and OSes, and same goes for app and channel, it is quite interesting to see only about 40,000 IP addresses are responsible for 10,000,000 ad clicks in less than a day! That means on average an IP address has about 250 ad clicks in a few hours! Clearly some IP have abnormally too many clicks, possibly click factories fraudsters.

This graph shows some IPs with the most number of empty clicks (positive):

![eda ip fraud clicks](assets/eda_ip_fraud_clicks.png)





### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_


Extreme Gradient Boosting (XGBoost) is an implementation of the gradient boosting machines that is highly flexible and versatile while being scalable and fast. XGBoost works with most regression, classification, and ranking problems as well as other objective functions; the framework also gained its popularity in recent years because of its compatibility with most platforms and distributed solutions like Amazon AWS, Apache Hadoop, Spark among others.

In short, XGBoost is a variation of boosting - an ensemble method algorithm that tries to fit the data by using a number of "weak" models, typically decision trees. The idea is that a "weak" classifier which only performs slightly better than random guessing can be improved ("boosted") into a "stronger" one that is arbitrarily more accurate (source: Y. Freund, R. E. Schapire)(source: R. E. Schapire). Building on the weak learners sequentially, at every round each learner aims to reduce the bias of the whole ensemble of learners, thus the weaker learners eventually combined into a powerful model. This idea gave birth to various boosting algorithms such as AdaBoost, Gradient Tree Boosting, etc.

XGBoost is an example of gradient boosting model, which is built in stages just like any other boosting method. In gradient boosting, weak learners are generalized by optimizing an arbitrary loss function using its gradient.

XGBoost, as a variation of boosting, features a novel tree learning algorithm for handling sparse data; a theoretically justified weighted quantile sketch procedure enables handling instance weights in approximate tree learning.

source: T. Chen, C. Guestrin

There is a number of advantages in using XGBoost over other classification methods:

Work with large data: XGBoost packs many advantageous features to facilitate working with data of enormous size that typically can't fit into the system's memory such as distributed or cloud computing. It is also implemented with automatic handling of missing data (sparse) and allows continuation of training, or batch training which was a tremendous help for me in this project.
Built-in regularization: XGBoost supports several options when it comes to controlling regularization and keeping the model from overfitting, including gamma (minimum loss reduction to split a node further), L1 and L2 regularizations, maximum tree depth, minimum sum of weights of all observations required in a child, etc.
Optimization for both speed and performance: XGBoost provides options to reduce computation time while keeping model accuracy using parallelization with multi-core CPU, cache optimization, and GPU mode that makes use of the graphics unit for tree training.







### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_



A benchmark for this problem is a classifying model that randomly guess whether or not a registered click on an advertisement would result in a download of the advertised app. The guessed probability is either 0 or 1. On the testing dataset, the model scored 0.3893 of the ROC-AUC metrics measuring the closeness of the resulted probability graph versus that of the solution.



## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_



As previously mentioned, the given dataset had already been well-prepared and processed, all categorical features had been labelled and set to numbers, sensitive data encoded, and potentially missing data had been either filled or discarded.

Data conversion
Before being feed to the model, the time attributes need to be converted to a format that XGBoost can understand. In its raw form, the attribute click_time was given in the format of yyyy-mm-dd hh-mm-ss; I have chosen to extract this into 3 new features: click_day, click_hour, click_minute, corresponding to the day, hour, and minute the click was registered, and in integer format. For click_time, the year and month information was disregarded simply because the data was recorded in the same month, so these values should be the same for all data points. On the other hand, I did not extract the second out of click_time because this value by itself does not how much meaning; in my opinion, humans are precise creatures but not down to the seconds in most cases. Moreover, I later made use of the click second in an engineered feature.

Feature engineering
How many clicks does an user make to a certain app?
Between a normal mobile user who clicks on an advertisement out of personal interest and a fraudster whose aim is to make as many empty clicks as possible, the fraudster would make many more clicks on the same advertisement than the normal user. This is because in a typical case once the user have visited the ad, he/she would be able to make the decision immediately or after just a few more clicks unless he/she is very indecisive! Therefore, a fraudster would make hundreds or even thousands of clicks on the same ad compared to a handful by an user, and this could be a clear distinction for the tree classifiers to pick up.

In order to create this new feature, avg_app_click_by_ip, I grouped the data based on the app, and divided the number of clicks by the number of unique IP addresses. Why did I only consider IP addresses and not device, OS, and channel? This is because a common "click farm" where fraudsters mass produce clicks can be set up with various devices and operating system, as can be seen here. In recent years, mobile frauds seem to have evolved to be more sophisticated in order to avoid the countermeasures put up by authorities. They equip themselves with varying phones and tablets models in order to mislead advertisers into thinking that several users are interacting with their advertisements. However, it is not easy for fraudsters to hide or relay their IP addresses because in countries such as China where mobile frauds are prevalent, it is not easy to access services such as VPN or relay servers.

Is a given click a one-off event or the user actually is on a click-spree?
As the goal is to tell clicks by genuine users from those by fraudsters who typically make a lot of clicks but never actually download apps, I think it is important to know how long does it take for the same user to make another ad click. The intuition is that a genuine user generally takes longer before making another click, since he/she would need time to decide whether to download the app or not, while a fraudster who never intends to download the app in the first place would not need to do so and hence takes less time before clicking another ad.

This new feature, time_to_next_click, is the amount of time measured in seconds in between clicks made by devices of the same IP address, model, OS, and from the same channel. While this can be a decisive feature differentiating a fraudster and an ordinary user, it is clear that it is not perfectly accurate due to the previously mentioned fact that the given data cannot exactly pinpoint individual users; it is not uncommon to have users with the same device and OS on the same network in school, cafe, or building, and these people may be wrongly taken as the one user.

In the end however, due to the sheer size of the data being too large, Pandas was unable to fully construct this feature on the whole training dataset of nearly 180,000,000 lines, so I decided to construct it on a portion of the data only.





### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_


he software requirement for the implementation is as followed:

Python >= 3.6
numpy >= 1.14.3
pandas >= 0.23.0
scikit-learn >= 0.19.1
xgboost == 0.72
I first attempted to train an XGBoost model without the engineered features, then planned to compare the performance to the same model trained with data with added features. The model's performance is evaluated by its ROC-AUC score on the testing data. The model's parameters were as followed:

Objective function: logistic binary
Scale positive weight: 99 (scale up the weight of the positive data points is_attributed==0 to counter the imbalance of the data)
Parallel jobs: 3 (to make use of CPU cores)
Tree method: "exact" On a system with a 4-cored CPU and 16GB of RAM, the dataset itself occupied half of the system memory, therefore XGBoost encountered memory error training on the whole dataset. Moreover, constructing time_to_next_click feature was also impossible due to memory shortage. This data proved to be simply impossible to train as it is. At this point I decided to use on only a portion of the data; at 10 million data points, the model scored a point of 0.5445 on the test data, significantly higher than the baseline of 0.3893 by the random-guessing model, but is still a terrible score. There was much to be done to refine this result.







### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


Batchnorm, regularization



Hyperparameter tuning
The refinement process started with hyperparameter tuning. Thanks to XGBoost's versatility that its classifier object XGBClassifier is actually compatible with scikit-learn framework, I was able to make use of scikit-learn's GridSearchCV object to perform tuning. The search space contains values for the following parameters:







## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_


Even though I managed to bypass the system memory limit on XGBoost by training the model incrementally, it still remains that Pandas is unable to produce the engineered feature time_to_next_click on the whole dataset. Therefore, I decided to again train the model on just a subset of the data. After some trials and errors, I found that 60,000,000 was the limit to which Pandas managed to generate time_to_next_click without memory error.

The final model design was built with the tuned hyperparameters as mentioned earlier, using GPU mode (tree_method: gpu_hist), trained on 60,000,000 data points with avg_app_click_by_ip and time_to_next_click features along with extracted time features. Score obtained on the testing set: 0.9700

To see the effect of having a smaller dataset but one extra feature, I have also trained another model with the same hyperparameters and on the whole training dataset, but without the time_to_next_click feature. Score obtained on the testing set: 0.9575

In conclusion, having time_to_next_click improved the performance of the dataset (0.0125) over having just the avg_app_click_by_ip feature even when trained on just about 30% of the data. It is certain that when trained with the whole dataset, this feature would increase the model's accuracy even more.



### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_




The final model design with tuned hyperparameters trained on a third of the data with extra features scored 0.9700 on the testing data, dwarfing the score of the random guessing model at 0.3893. This means that the final model far surpasses the random guessing model in terms of learning the target concept, as a perfect predictor would give a score of 1.0.

In the boundaries of the competition, I can say these results are encouraging and show that the approach taken is the right direction, although there is so much to improve upon.




## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_


To get a glimpse, I constructed more time_to_next_click features, this time using different aggregations of given features [ip, os, device, channel, app] to identify individual "clicking session" or "clicking spree". As previously mentioned, the intuition is that a fraudster would have very short amount of time in between advertisement clicks compared to a normal mobile user. I also paired ip with app and channel to construct time to next click features, because as mentioned earlier some fraudsters may make use of "click farm" with various different devices to generate fake clicks, and in those cases the IP address is the only way to identify them.

Again due to hardware limitation, I was only able to construct the new time_to_next_click features with 10,000,000 rows of the training data. I then fitted the data with an XGBoost classifier and used the plot_importance feature of XGBoost to see how much each feature has an impact on predicting the outcome of a click. The graph of feature importance is given below:

![feature importance](assets/feature_importance.png)
As can be seen, among the 7 newly constructed time_to_next_click features, groupings of [ip, app] and [ip, os, device, app] were the most impactful. This aligned with my above assertion that for some cases, [ip, app] is the best way to identify a fraudster's "clicking session"; [ip, os, device, app] on the other hand shows that a fake click can still be identified with more specific user information aside from just their IP address; perhaps these are cases where fraudsters use mostly similar devices because they were cheap?




### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_


End-to-end problem solution
This project turned out to be one that emphasizes on handling big data and utilizing computational resources; the given data was clean and thoroughly processed, the target concept is clearly defined, the features gave lots of room for engineering and research. All that was left was working around the sheer gigantic size of the data and optimizing all computing power at hand to fit it in, so much so that the data itself became the main problem instead of the target concept throughout the project.

Therefore, my problem solution not just includes data processing and training but also hardware optimization:

Establish basic statistics and understanding of the dataset such as imbalance, data type, etc.
Data cleaning was not needed as the given data was thoroughly processed by TalkingData.
Extracted clicking time information (day, hour, minute) into a format usable by trainer.
Devise new features based on the given features:
Average click on the same advertisement by the same IP address
How much time does it take until the same IP address clicks again?
This feature proved to be impossible to produce on the whole training data due to hardware limitations
Train and test model's performance:
Model was unable to fit the whole dataset, again due to hardware limitations
As time_to_next_click was also impossible to produce on the whole data, used only 10 million data points instead
Score: 0.5445
Improve:
Fine-tune model's parameters with GridSearchCV
Implemented incremental training so that model can make use of the whole dataset
Setup model to train on GPU, computation time improved by 75%
Train and test again:
time_to_next_click was still impossible to produce, so I decided to use 60 million rows of data this time; it is small enough so that Pandas can generate the feature. Score: 0.9700
To compare, I trained another model with the same hyperparameters, but with all the data and without the time_to_next_click feature. Score: 0.9575



### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_





There is so much that can be improved upon for this project, for example the same model design training on the whole dataset with all generated features present would no doubt yield a much higher performance.

Overcome memory limits for feature engineering
I was unable to generate time_to_next_click feature on the whole dataset due to hardware limitations. After doing some research, I found that there were ways to workaround this such as using distributed system and paging memory.

As I understand, distributed systems such as AWS clusters help overcome the memory constraint by distributing the data into its many node, each holds a chunk of the data that can work as a whole. These systems utilizes parallelization for speed and computing power, and definitely would be a big help in problems that deal with big data like this one.

Another potential workaround that I found was using in-storage memory. Apparently, it was possible to construct certain file formats that resemble how data is stored on RAM, but instead these files can be stored on the system's hard drive which typically has more generous size than RAM. This method would the training data to be accessible by the system, without the need to have a big enough memory. This method may also obsolete the incremental step of training process, since the whole data is readily available at all time for the model to access.

More time features worth exploring
My intuition is to follow the clicking patterns of individual users, and the time between the clicks has proven to be an important feature. To exploit this further, we can also generate more time-between-click features such as how much time does is it between this click and the next 2 clicks, or how much time between this click and the previous click. The reason is that for normal users, there may be cases where the user makes more than 1 click in a short amount of time due to accident or because the user realizes something after finishes reading the advertisement, and goes back to it. These cases may be falsely identified as fake clicks, so to be able to look 2 or more clicks ahead would be helpful since it would be more rare for normal users to generate repeated clicks in any case.

More app-based features
Aside from time features, app-based and channel-based features may hold important information, since intuitively a fraudster often target a single app or advertisement to generate fake clicks, and often operates exclusively in certain app distributing channels since some channels do not have strict regulation against mobile frauds.

Similar to the time features, app and channel can be associated with individual users or "clicking sessions" by aggregating them with different groups of given identity features: [ip, device, os]. Upon successfully generating these new features, I would gauge their importance similar to what I did in the previous section and include the 2 most important ones for the final model training.




-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
