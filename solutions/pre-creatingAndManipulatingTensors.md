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

# creating_and_manipulating_tensors

## Exercise 1: Introduce a Third Operand
```python
# Write your code for Task 1 here.
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
print("primes: ", primes)

squared_primes = tf.multiply(primes, primes)
print("squared_primes:", squared_primes)

one = tf.constant(1, dtype=tf.int32)
just_under_primes_squared = squared_primes - one 
print("just_under_primes_squared:", just_under_primes_squared)
```

## Exercise 2: Reshapre two tensors in order to multiply them.
```python
# Write your code for Task 2 here.

# Matrix A is 1 x [6] 
# Matrix B is [1] X 3
a = tf.constant([5, 3, 2, 7, 1, 4])
b = tf.constant([4, 6, 3])

# so lets make a equal 2 x 3 
# and have B equal 3 x 1 (so swap them)
reshaped_a = tf.reshape(a, [2,3])
reshaped_b = tf.reshape(b, [3,1])

solution = tf.matmul(reshaped_a, reshaped_b)
print(solution)
```
Corey Ford 
