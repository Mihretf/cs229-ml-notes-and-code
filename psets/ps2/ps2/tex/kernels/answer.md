# Kernel functions: 
instead of explicitly doing the exhausting math to transform our data into massive, high dimensional spaces like mapping trig function or pdcts, a kernel calculates the similarity between two data points using only their original inputs.
Postitive semi-definite: a matrix is PSD if multipying it by any vector z in a quadratic form never results in a negative number, it always evaluates to zero or a postive number, representing a stable, non-negative "energy" state. 
Mercer's Theorem: it states that any function is a legally valid kernel if and only if its resulting similarity matrix is completely symmetric and positive semi-definite for any set of points. 
