```
Spam classification


- Naive Bayes model: probabilistic classification algorithm built on Baye's Theorem. It is designed to answer fundamental questions: "Given a set of clues or features that we observe, what is the probability that this item belongs to a specific category or class?"

It models how data is generated for each class by looking at the joint probability distribution P(x,y)
No iterative Training- unlike linear or logistic refression, we don't need to run thousands of looping optimization steps to find weighs, training this model is bascially just going to be counting things and finally calculating the probabilties directly. It like uses logs add instead of multiplying tiny decimals, which means in can train and predict on massive datasets almost instantly. 


1. Data Preprocessing and vocabulary building
Before a model can look at text, computers only understand numbers. We convert text messages into a structured format through 3 stages
a. Getting words
b. Creating the Dictionary
c. Transforming text into a matrix- this is by creating a massive grid or matrix filled with zeros. Rows represent individual text messages, and the columns represnet every valid word in our dictionary. For every message, it checks each word. If the word exists in our dictionary, if finds its column index j and adds 1 to matrix[i,j], counting how many times that word appeared in that soecific message. 
```
# 2. Training the Naive Bayes Model

num_messages, vocab_size = matrix.shape
- We are extracting the dimensions of our training matric, m is total number of messages or rows and V is the total vocabulary size or columns. This means we are sizing up our dataset, because we need to know how many total texts we are looking at and how many unique words our dictionary tracks so we can loop through and count properly. 

```
> num_spam = np.sum(labels ==1)
>p_y_is_1 = num_spam /num_messages
>p_y_is_0 = 1- p_y_is_1

```
So we are calculating probaility that y is 1 and p of y being zero, which is 1 minus prob of y being 1 
P(y=0) = 1 - P(y=1)
- We calculate our baseline odds. We count how many messages are labeled as spam and divide by the total number of messages to find out what percentage of our training data is spam overall. 

```
>spam_matrix = matrix[labels == 1]
>non_spam_matrix = matrix [labels == 0]

```
We are filtering our rows of our feature matrix based on boolean conditions corresponding to the target label vector. 
Basically, we are sort of piling our mail. We separate all our text records into two distinict buckets: one bucket holding only spam messages, and other holding only regular message, the labeling is done if label has one then it will be sorted inside the spam matrix, if the label is 0 then it will be for the non spam matrix. 

```
> alpha = 1.0
> spam_word_counts = np.sum(spam_matrix, axis =0 )
> total_spam_words = np.sum(spam_word_counts)

>non_spam_word_counts = np.sum(non_spam_matrix, axis =0)
>total_non_spam_words = np.sum(npn_spam_word_counts)

```

We are computing column wise sums to get raw frequency vectors for each class, and scalar sums for total word counts. alpha sets the laplace smoothing parameter making it 1. 
? What is laplace smoothing, what does it mean and what is it doing?

What it means: It acts as a baseline safety cushion by adding a small, uniform pseudocount (typically $+1$) to every single feature or word count.Why it matters: Without it, if a brand-new or rare word appears during testing that was never seen in the training data, its probability evaluates to zero, which can wipe out or crash an entire multi-word probability calculation. Because we would need to multiply everything later on. 

Inside each bucket, the spam and non-spam, we are counting up how many times every single word appears. Example, we find out exactly how many times the word "free" shows up across all spam emails combined. 

```
# p_word_given_spam = (spam_word_counts + alpha) / (total_spam_words + alpha * vocab_size)
# p_word_given_non_spam = (non_spam_word_counts + alpha) / (total_non_spam_words + alpha * vocab_size)

we are applying laplace smoothing, we are calculating the chance of seeing a specific word given a class. We add a tiny fake count (alpha) to everything so that if a brand-new word pops up later that we've never seen before our math will not break. 

We are sort of finding the p of x would be a specific word of w given that y is 1.

# model = {
   # 'p_y_is_1': p_y_is_1,
   # 'p_y_is_0': p_y_is_0,
   # 'log_p_word_given_spam': np.log(p_word_given_spam),
   # 'log_p_word_given_non_spam': np.log(p_word_given_non_spam)
}
# return model

We wrap probabilites in natural logs to convert tiny decimal multiplications into sage additions, preventing underflow and package everything into a dictionary. We need to change them into logs because when multipling small decimals they will make our computers panic. 

```
# Predicitng from the naive bayes model 

We just pull up our previous log-priors and conditionals from the dictionary
? What is joint log posterior,

We then check whether spam_score > non_spam_score, evaluating then giving boolean array, either true or false. Then if the spam score is greater, then that message is spam, and if is less than, then that message is not spam. 
