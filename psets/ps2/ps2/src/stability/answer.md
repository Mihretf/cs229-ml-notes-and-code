```
1.  Logistic Regression: Training stability

a.
 The data set A training model took 30371 iterations and converged, finished training the mode. The data set B traning model did not converge, it went on and on until stopped automatically, at minumum it went through 1160000 iterations before I stopped it. The theta does not converge it just keeps on increasing and just going and going. 

b. 
The reason this happens is: 
On Dataset A: which is the linealy inseparable one, since we dont have a specific line that will classify these points, we will need to find the best possible number or the best possible weight that will help us to classify the data plots. And because there is a specific value for which we will get the best optimal point, our training successfully converges. 
On Dataset B: here however we have linerally separable points, which means that we can continue to adjust our weight knobs again and again, increasing it and going up to infinity, because it is separable there will come a better, bigger number that will sort of satisfy the property of our classification, so it will just increase and increase, it will never converge, so we sort of are not able to learn successfully from that. 

- Changing a constant learning rate only changes the speed of the gradient steps, not the ultimate destination. So, it would still go upto inifinty, it will not converge. 

- Decreasing the learning rate over time: if the learning rate shrinks fast enough, the step sizes shrink to zero faster than the gradients push tetha outward, so it allows the tetha to freeze at a finite value. 

- Linear scaling of the input features: multiplying the input features by constant factor just stretches or compresses it but the weight will still diverge to infinity. 

- Adding a regularization term to the loss function: yes it would converge, because L2 regularization penalizes large parameter weights. It changes the objective function to include a cost for magnitude, forcing the optimization to balance minimizing classification error with keeping the tetha small, giving us a global minimum. 

- Adding zero-mean Gaussian noise to the training data or labels: if we add noise it would converge because adding noise blurs the strict boundary between classes and introduces overlapping points, destroying the perfect linear separability. 

- Support Vector Machines: they are supervised machine learning models used for classifiction and sometimes regular tasks as well. 

These machines do not stop once they find any line or hyperplane that separates the classes, which is what standard linear models do, but they explicitly search for the optimal decision boundary that maximizes the margin. 
The final decision boundary is entirely determined by as subset of the training data points known as support vectors. 
They can also efficiently learn non-linear decision boundaries by using feature maps with kernel functions, this allows them to project data into higher dimensional spaces where complex patterns become lineraly separable without explicity computing the high dimensional coordinates. 
By maximizing the margin and utilizing regularization, SVMs provide strong theoretical guarantees against overfitting, even in high-dimensional feature spaces. 