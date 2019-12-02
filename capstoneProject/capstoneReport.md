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

The data was acquired from the [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection/overview) `Kaggle` competition. 

### Data files

- [`training.csv`](https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip): list of `7049 training images`. Each row contains the (x, y) coordinates for 15 keypoints, and image data as row-ordered list of pixels.
- [`test.csv`](https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip): list of `1783 test images`. Each row contains ImageId and image data as row-ordered list of pixels

### Problem Statement

The objective of this project is to accurately predict the facial keypoints (facial landmarks) of a face image. My hypothesis is, that this prediction can be performed based on a training set containing accurate facial keypoints, through a regression approach.

A `Convolutional Neural Network` (`CNN`) will be applied to predict the facial keypoints. A `CNN` was chosen for this problem because:

- This is a computer vision problem that requires capturing features for prediction
- CNNs are very useful in capturing features in images
- The expected responses (coordinates) make this a regression problem

_A simple `fully connected network` will be used as a baseline model for comparison._

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

Each predicted keypoint is specified by an (x,y) real-valued pair in the space of pixel indices. There are 15 keypoints, which represent the following elements of the face:

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

### Exploratory Visualization

The data is summarized as follows.

```python
Data columns (total 31 columns):

left_eye_center_x            7039 non-null float64
left_eye_center_y            7039 non-null float64
right_eye_center_x           7036 non-null float64
right_eye_center_y           7036 non-null float64
left_eye_inner_corner_x      2271 non-null float64
left_eye_inner_corner_y      2271 non-null float64
left_eye_outer_corner_x      2267 non-null float64
left_eye_outer_corner_y      2267 non-null float64
right_eye_inner_corner_x     2268 non-null float64
right_eye_inner_corner_y     2268 non-null float64
right_eye_outer_corner_x     2268 non-null float64
right_eye_outer_corner_y     2268 non-null float64
left_eyebrow_inner_end_x     2270 non-null float64
left_eyebrow_inner_end_y     2270 non-null float64
left_eyebrow_outer_end_x     2225 non-null float64
left_eyebrow_outer_end_y     2225 non-null float64
right_eyebrow_inner_end_x    2270 non-null float64
right_eyebrow_inner_end_y    2270 non-null float64
right_eyebrow_outer_end_x    2236 non-null float64
right_eyebrow_outer_end_y    2236 non-null float64
nose_tip_x                   7049 non-null float64
nose_tip_y                   7049 non-null float64
mouth_left_corner_x          2269 non-null float64
mouth_left_corner_y          2269 non-null float64
mouth_right_corner_x         2270 non-null float64
mouth_right_corner_y         2270 non-null float64
mouth_center_top_lip_x       2275 non-null float64
mouth_center_top_lip_y       2275 non-null float64
mouth_center_bottom_lip_x    7016 non-null float64
mouth_center_bottom_lip_y    7016 non-null float64
Image                        7049 non-null object

dtypes: float64(30), object(1)
```

Also, below is an example group of images.

![sample image](./assets/sampla_data.png)

### Algorithms and Techniques


### Benchmark

In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_

## III. Methodology

### Data Preprocessing

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation

In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_


The software requirement for the implementation is as followed:

```python
Python >= 3.6
numpy >= 1.14.3
keras >= 
tensorflow >= 
pandas >= 0.23.0
```

### Refinement

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_

## IV. Results

### Model Evaluation and Validation

In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification

In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_

## V. Conclusion

### Free-Form Visualization

In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection

In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement

In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
