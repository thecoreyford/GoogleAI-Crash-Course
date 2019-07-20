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

--

# Descending into ML
**Linear regression** is a method for finding the strait line (or hyperplane) of best fit.

**y=mx+c** is instead refered to as **y=wx + b** with **w** meaning **weights** and **b** meaning **bias**. 

A subscript may be used as we could be in more than 1 dimenson.

Loss is showing how well our line is doing at predicting any given example (different between prediction and true value for example).

### A convenient loss function: L<sup>2</sup> Loss (or squared error)
* square of the difference between prediction and label 
* (observation - prediction)<sup>2</sup>
* (y - y<sup>'</sup>)<sup>2</sup>




