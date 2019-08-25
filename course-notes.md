# Framing
## Terminology:
* **Label** - the thing we are predicting (i.e. y in linear regression).
* **Feature** - is the input variable (i.e x in linear regression). May have multiple.
* **Examples** - a particular instance of data:
    * **Labeled Example** - includes features and the label. Use this to train (i.e. label could be "spam" or "not spam")
    * **Unlabeled Example** - includes just features - can be used to predict the label on unlabeled examples. 
* **Models** - relationship between features and label
    * **Training** - Showing the model labeled examples for it to learn relationships between features and labell.
    * **Inference** - applying trained model to unlabelled examples.
* **Regression** - predicts continous values ("what is the vaule of a house?" - "probability user will click on this add?")
* **Classification** - predicts discrete values ("Is this email spam?" - "is this a dog, cat or chicken?")

## Other notes:

Think about label reliability!

Useful features are quantifiable - subjective non measurables such as "adoration" are not!

---

# Descending into ML
## Linear Regression
**Linear regression** is a method for finding the strait line (or hyperplane) of best fit.

**y=mx+c** is instead refered to as **y<sup>'</sup>=w<sub>1</sub>x<sub>1</sub> + b** with **w** meaning **weights** and **b** meaning **bias**.
* *y<sup>'</sup>* is the predicted label (a desired output).
* *b* is the bias (y-intercept), sometimes w<sub>0</sub>.
* *W<sub>1</sub>* is the same as "slope" **m**. 
* *x<sub>1<</sub>* is a feature (input variable).  

A subscript may be used for more than 1 dimenson.

Loss is showing how well our line is doing at predicting any given example (different between prediction and true value for example).

### A convenient loss function: L<sup>2</sup> Loss (or squared error)
* square of the difference between prediction and label 
* (observation - prediction(x))<sup>2</sup>
* (y - y<sup>'</sup>)<sup>2</sup>

Likely to want to do this over a whole data set (sum - somethimes averaged)! I.e. for 3 values:
* y' = b + w<sub>1</sub>x<sub>1</sub> +  w<sub>2</sub>x<sub>2</sub> + w<sub>3</sub>x<sub>3</sub>

## Training and Loss
**Exmperical risk minimization** is building a model by examining many examples and finding a model that minimizes loss.

**Mean square error (MSE)** is the average squared loss per example over the whole dataset.
* Sum all the squared losses per example and divide by no of examples.

*Although MSE is commonly-used in machine learning, it is neither the only practical loss function nor the best loss function for all circumstances.*

---

# Reducing Loss 

### How do we reduce loss?
* **Hyperparameters** are configuration settings used to tune how the model is trained. 
*  Derivative of (y-y<sup>'</sup>) with respect to the weights and biases  (using y=wx+b) tells us how loss changes for a given example.
    * simple to compute and convex
* Hence, repeatably take small steps in direction that minimises loss 
    * called **gradient steps** (but really negative gradient steps.
    * Strategy is called gradient descent
* **Learning Rate** (sometimes step size) is how much you step through the curve (often scalar).
Plot Loss against *some variable* to help visualise and explain this!

*Convex* curves (live a bowl) will only have one minimum so the weights can start anywhere!
*Non-convex* curves (like egg crates) will have more than one minimum, so their is a strong dependency on initial values [neural nets are notoriously like this!!].

* **Stochastic Gradient Descent:** one example at a time
* **Mini-Batch Gradient Descent:** batches of 10-1000
    * Loss & gradients are averaged over the batch

## An Iterative Approach
Like going hot hot hot colder colder hotter hotter ... 

![iterative approach to training model](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg)

This keeps happening until rate of change is slow (or lowest possible loss))!

## Gradient Descent 

* When there are multiple weights, the gradient is a vector of partial derivatives with respect to the weights.
*Note that a gradient is a vector, so it has both of the following characteristics:*
    * a direction
    * a magnitude

As far as I understand the steps esensially are:
1. Knowing what the loss is, pick a value for a parameter: i.e. *w<sub>1</sub>*
2. Figure out the gradient of this point. This is the *derivate* (or **"m"**).
3. As the gradient always points towards the steepest increase in loss function, get the negative gradient.
4. Then add *some fraction* (scalar) to the gradient to move closer to the minimum. 

## Learning Rate
```c++
if (step size is too small) {
    //takes forever 
} else if (step size is too big) {
    // will jump around and overshoot the minimum
} else {
    // goldilocks
    // related to how flat the loss function is 
    // if gradient is small can try a larger learning rate!
}
```
**BONUS:**
The ideal learning rate in one-dimension is (the inverse of the second derivative of f(x) at x).

The ideal learning rate for 2 or more dimensions is the inverse of the Hessian (matrix of second partial derivatives).

## Stochastic Gradient Descent

*Batch* is the total number of examples used to calculate the gradient in a single iteration.

Large data sets would take ages to compute, however.

Lets get the right gradient on *average* through choosing examples at random from our data set. 

**Stochastic gradient descent (SGD)** takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration.

**Mini-batch stochastic gradient descent (mini-batch SGD)** is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples.


---

# First Steps With TF 

Tensor Flow is a General graph based API. 
    
This focuses on the tf.estimator API whitch is a high level abstraction making common tasks easier.

Tensflow consists of the following two components:
* a **graph** protocol buffer
* a runtime that executes the (distributed) graph
Kinda like how the python interpreters runs python code (the graph is executed by the runtime.)

**NOTE:** this is for many platforms (run on CPU, GPU, and TPU).

Works with *Scikit-learn* which is an open source ML python library.

---

# Generalization

* Do not overfit -> new data may overstep the "fancy" line
    * A model should be as simple as possible, so that a good empirical result is not due to peculiarities in our sample
* Empiricall Strategys
    * Training-set and Test-set ("Test Set methodology")
* Pull from the same distribution!!!!
    * UNLESS... fashions change - distibution may have to change
    * distributions may change over time also (e.g. shopping seasons change)
    
    
---

# Training and Test Sets

* Divide one set into one for training and one for test.
* Do a randomize!


* Larger training set the better the model will learn. 
    * but....
* Larger the test set the better confidence in the evaluation metrics.
    * A large data set will mean we can have good confidence intervals with smaller test set.
* **Do not train on test data! Double check before throwing a party!**


* Doing many rounds of this procedure might cause us to implicitly fit to the peculiarities of our specific test set.

---

# Validation Set 

* By partictioning the data set into 3 sets
    * Test set 
    * Training set
    * and Validation set
* ...you can ensure that you don't overfit to a single "test set"

* Meaning, evaluate model with a *validation set* and confirm the results with the *test set*.

---

# Representation

* **Feature engineering** turing raw data into a feature vector. 
* **One(multi)-hot encoding** is using a binary vector of length of categorical features, where applicable elements are set to "1".
    * If this is for a large number of categories you can use a [**sparse representation**](https://developers.google.com/machine-learning/glossary/#sparse_representation). 


## Qualities of good features

* **Avoid rarely used discrete feature values** (appear ~than 5 times)
* **Prefer clear and obvious meanings** (i.e. house_age: 1234538 is horrific)
* **Remove error data** such as "-1" appearing when a user dosen't awnser a question
    * *Instead add a question_not_defined param*
* **Avoid a feature name changing over time** 

## Cleaning data

* Scale values (likey to smaller floating point numbers)
* Handle extreme outliers
    * Clip values 
* Bin values between a < value <= b 
* Scrubbing 
    * ommited values, duplicates, bad labels, human errors
    
---



