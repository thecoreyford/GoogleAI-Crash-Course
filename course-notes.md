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









