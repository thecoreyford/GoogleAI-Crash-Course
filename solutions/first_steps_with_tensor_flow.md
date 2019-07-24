--------------------------------------------------------------------------------------------------
#### Copyright 2017 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

--------------------------------------------------------------------------------------------------

# first_steps_with_tensor_flow

**Catagorical data** not numbers, text descriptive

**Numerical data** int or floats!

**To pull thing from dataframe:**
```python
targets = california_housing_dataframe["median_house_value"]
```

**Stolen from step 3:**
```python
# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)#this uses mini-batch stochastic GD
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)#clips gradients so not too large!

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor = tf.estimator.LinearRegressor(
feature_columns=feature_columns,
optimizer=my_optimizer
)
```

**Epochs** means the whole data, not just a batch step!

###### Exercises mostly consisted of tweaking parameters!

Corey Ford 
