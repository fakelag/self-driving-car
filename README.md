# Self Driving Car

Self driving car with steering angle prediction using a deep neural network

## CNN Layout
| Layer (type) | Output Shape | Param #  |
| ------------- | ------------- | ------------- |
| conv2d_1 (Conv2D) | (None, 31, 98, 24) | 1824 |
| conv2d_2 (Conv2D) | (None, 14, 47, 36) | 21636 |
| conv2d_3 (Conv2D) | (None, 5, 22, 48) | 43248 |
| conv2d_4 (Conv2D) | (None, 3, 20, 64) | 27712 |
| conv2d_5 (Conv2D) | (None, 1, 18, 64) | 36928 |
| flatten_1 (Flatten) | (None, 1152) | 0 |
| dense_1 (Dense) | (None, 100) | 115300 |
| dense_2 (Dense) | (None, 50) | 5050 |
| dense_3 (Dense) | (None, 10) | 510 |
| dense_4 (Dense) | (None, 1) | 11 |

## Learning loss with current training data
![Learning loss](./img/l1.png?raw=true)

## Simulator used
https://github.com/udacity/self-driving-car-sim
