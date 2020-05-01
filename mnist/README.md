# template.cpp

In order to understand the basic concept of neural networks, I decided to attempt the MNIST problem in C++.

My main resource for this was the 3b1b neural network series. It was really comprehensive and I highly recommend it.

Is this any good? It's mediocre. In fact the implementation is extremely slow / suboptimal. After all instead of matrices, typical arrays / vectors were used. Optimisation could considerably accelerate this.

Nonetheless it was an excellent learning experience, and my understanding of all this is much more intuitive. Hopefully the code is reasonably self-documenting, or I have failed in my task.

The file `net` contains the network weights and biases themselves. These can be read by calling `populate(net)`.

Will I try reusing this implementation? Obviously. Probably find a decent linalg library too.

[Wanderer's Lullaby](https://www.youtube.com/watch?v=70VlAyEUXYM)

**Run `./template net` to train it. Dependencies: train-images and train-labels in directory.** 

# test.cpp

This uses the testing dataset to get the net's accuracy. (The train function is silent because printing is slow.)

**Run `./test net` to test it. Dependencies: test-images and test-labels in directory.**

# attempt.cpp

This uses OpenCV to read in an image, and predict the digit. Resizing *should* occur on its own...

Compiling: `cmake .; make` should create the file `attempt`.

**Run `./attempt net img1.png img2.png ...` and predictions are printed as output.**