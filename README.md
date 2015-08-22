Two Layer Neural Network in Python
================
Created on: 08/22/2015

## Synopsis

This is a class for two layer neural networks written in Python. This class was used to obtain 96% accuracy on MNIST data.


## Code Example

```python
from nn import TwoLayerNeuralNetwork
my_nn = TwoLayerNeuralNetwork(n_in=784, n_hid=200, n_out=10, eta=.01, epochs=5) #Initialize NN
my_nn.train(train_images, train_labels, validation_images, validation_labels, save_weights=True) #Train NN
pred = nn.predict(test_images) #Predict Test images
```

## License

The MIT License (MIT)

Copyright (c) 2015 Shane Barratt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.