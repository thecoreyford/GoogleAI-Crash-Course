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

# tensorflow_programming_concepts

## Exercise: Introduce a Third Operand
```python
# Create a graph.
g = tf.Graph()

# Establish our graph as the "default" graph.
with g.as_default():
# Assemble a graph consisting of three operations. 
# (Creating a tensor is an operation.)
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    z = tf.constant(4, name="z_const")
    my_sum = tf.add(x, y, name="x_y_sum")
    new_sum = tf.add(my_sum, z, name="x_y_z_sum")

# Now create a session.
# The session will run the default graph.
with tf.Session() as sess:
    print(new_sum.eval())
```

## Exercise 2:
```python
cities.reindex([0, 4, 5, 2])
```
Corey Ford 
