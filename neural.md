# Neural Networks! (the CS module I never took)

## Cool things to search up

**Generative Adversarial Networks (GANs)**: a 'generator' and a 'discriminator' network that essentially fight each other

**Cross-entropy loss function**

**Learn numpy**

**Genetic algorithms**

## K-nearest neighbours

Does this really need to be explained? I mean, come on.

There's probably a way to optimise this, but 

---

## Feedforward neural networks (watch: the 3b1b series)

You have a system of nodes.

- Edge weight: $w_{layer, target,source}$
- Bias: $b_{layer,node}$
- Value: $a_{layer, node}$
- $f(x)=\max{(x,0)}$
- thus. $z_{l,i}=b_{l,i}+\sum_{j=0}^kw_{l-1,i,j}a_{l-1,j}$
- $a_{l,i}=f(z_{l,i})$


$$
\begin{align}
\frac{\partial a_{l,i}}{\partial z_{l,i}}=f'(z_{l,i})&=\begin{cases}1 &z_{l,i}>0\\0 &z_{l,i}\le 0\end{cases}\\\frac{\partial a_{l,i}}{\partial b_{l,i}}=\frac{\partial a_{l,i}}{\partial z_{l,i}}&=\begin{cases}1 &z_{l,i}>0\\0 &z_{l,i}\le 0\end{cases}\\\frac{\partial a_{l,i}}{\partial w_{l-1,i,j}}&=\begin{cases}a_{l-1,j} &z_{l,i}>0\\0 &z_{l,i}\le 0\end{cases}\\\frac{\partial a_{l,i}}{\partial a_{l-1,j}}&=\begin{cases}w_{l-1,i,j} &z_{l,i}>0\\0 &z_{l,i}\le 0\end{cases}\\\frac{\partial C}{\partial a_{L,i}} &=2(a_{L,i}-x_i)\end{align}
$$
Thus flow:

- Test about 100 images. For each image calculate the nudge for each weight and bias.
- When this 'batch' is complete, actually 'step' by each nudge. This moves each value closer to minimising the cost function.
- Repeat this indefinitely.

### Important notes:

- The "learning rate" should be ~-0.01 to -0.1 ish. 
- Stochastic gradient descent: instead of using the entire dataset, you take the gradient across a batch of about 100 images. *Then* you 'step' everything by it, and go to the next batch.
- You should iterate through the whole batch several times. A lot of times.
- ReLU is convenient, but has a tendency to 'kill' neurons (once they reach 0 they stick at 0.) This is called the "dying ReLU problem". This can be fixed via a **leaky** ReLU which has a small positive gradient when the argument is negative.

### Okay, this is all well and good, but it's really slow. How do I exploit linear algebra instead?

Well, you could put all the weights in a matrix, I guess? And have the $a_{layer,i}$ things in a vector.

You can get a row of a matrix. And just... like, multiply them. Termwise. (Dot product I guess.)

But how much will it help? Each weight has to be updated individually anyway, so that bounds the time complexity.

Will probably look more into this later.

---

## Convolutional Neural Networks (CNNs)

These networks are optimised for **image processing** and should *not* be used if **swapping columns gives equally useful data**. (Neighbours!)

At their core is the operation **convolution**: application of a 'filter' (5x5 or similarly small) which measures how well a region of pixels 'matches' with a pattern, returning a double from 0 to 1 --> this returns you a matrix similar to the original so further filters can be added.

Value is typically $\sum{d_i^2}$ per pixel. It's kind of like a normal neuron, honestly. Just that there's a matrix of values instead of just one, so... **sad time complexity noises.** Hopefully your image is small. It's essentially O(image size * filter size).

The CNN comprises layers based on **convolution**, **pooling** (reduce the image size: done either by taking maximums or summating values. In principle this uses a filter size (say $2\times2$) and step size (say 2 steps).) and **ReLU** (generally after convolution, before pooling).

At the end, the remaining input is put through a **fully-connected network**. This is just the 'normal one'. It ends in the outputs you're looking for.

### Backpropagation

In principle this is actually not all *that* different from the fully-connected one.

- Pooling: easy to propagate (just add to all behind in range). Derivative is just 1 for max and 0 otherwise.
- ReLU: standard.
- Convolution: honestly, O(image size * filter size) is fine. Do it manually.

### Linalg exploitation?

Sad linalg noises.

### Implementation / flow?

One 'layer' would be a convolution-pooling-ReLU set, I guess...? So just store everything.

`node[layer][item][size]`

`filter[layer][item][item+1][size]`

and then at the end you just turn everything into nodes and run a normal NN on it

Just propagate the $da$.

Say you have $da$ on a layer.

Then backprop: $da$ to only max and ReLU-ed.

Then backprop: $da$ by convolution. For each thing, $2(output-filter)$

So $da$ is a variable on each level.

### Important notes

- The learning rate (`s`) is **very important**. If the values all go zero (killed by ReLU) the rate is too high. It should be obvious if it is too low.
- There's a command line option (`-fsanitize=address`)  that lets you check if the indices are out of bounds. Needless to say this is helpful.
- If the values might just average out (0.1 for everything) then... well I haven't actually solved this yet. **Help me.**

---

## Recurrent Neural Networks (RNNs)

The way this works is by feeding the previous output **into the input**, allowing simple predictions to be made.

## LSTM (Long Short-Term Memory)

Consider a 'typical' neural network. It reads in data and prior predictions, and outputs a new prediction.

The **input** is prior predictions + input data.

Now we add another neural network which reads the input and is trained on what to **ignore**. The output is multiplied by the prediction. Then the **memory** is added to the prediction.

Part of this prediction is now multiplied by another neural network which reads the input and is trained on what to **forget**. The output becomes the new **memory**.

Finally a **selection** network is multiplied to select only the relevant predictions.

### Why specifically these four?



