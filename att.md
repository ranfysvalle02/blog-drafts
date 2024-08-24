# Understanding Self-Attention Mechanism in Natural Language Processing

In this blog post, we will delve into the self-attention mechanism, a key component of many modern Natural Language Processing (NLP) models, such as the Transformer model. We will explore this concept through a Python code snippet that uses the self-attention mechanism to predict the next word in a sentence.

## The Code

The code begins by importing the numpy library, which provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

```python
import numpy as np
```

### The Softmax Function

The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or large, the softmax function squashes it, which helps in mitigating the exploding and vanishing gradient problems.

```python
def softmax(x):
  x -= np.max(x)
  exp_x = np.exp(x)
  softmax_x = exp_x / np.sum(exp_x)
  return softmax_x
```

### Creating Word Representations

The `create_word_representations` function takes a list of sentences as input and creates a dictionary mapping words to indices and vice versa. It also creates a list of word embeddings, which are randomly initialized.

```python
def create_word_representations(sentences):
    word_to_index = {}
    index_to_word = {}
    word_embeddings = []

    for sentence in sentences:
        for word in sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word[len(index_to_word)] = word
                word_embeddings.append(np.random.rand(3))  # Random embeddings

    return np.array(word_embeddings), word_to_index, index_to_word
```

### Calculating Self-Attention

The `calculate_self_attention` function calculates the attention scores for each word in the context. It then computes the attention weights by applying the exponential function to the scores and normalizing them.

```python
def calculate_self_attention(query, keys, values):
    scores = np.dot(query, keys.T) / np.sqrt(keys.shape[1])
    attention_weights = np.empty_like(scores)
    for i in range(len(scores)):
        if len(keys[i].shape) == 1:  # Check if 1D array
            attention_weights[i] = np.exp(scores[i])  # No need to sum for unique words
        else:
            attention_weights[i] = np.exp(scores[i]) / np.sum(np.exp(scores[i]), axis=1, keepdims=True)

    return attention_weights
```

### Predicting the Next Word with Self-Attention

The `predict_next_word_with_self_attention` function uses the self-attention mechanism to predict the next word in a sentence. It first calculates the context embeddings by averaging the embeddings of the words in the context window. It then calculates the attention weights and applies the softmax function to get a probability distribution over the words. The next word is predicted by sampling from this distribution.

```python
def predict_next_word_with_self_attention(current_word, context_window, words, word_embeddings, word_to_index, index_to_word):
    context_embeddings = word_embeddings[[word_to_index[word] for word in context_window]]
    query = np.mean(context_embeddings, axis=0)  # Average context embeddings
    keys = values = np.array([word_embeddings[word_to_index[word]] for word in words])
    attention_weights = calculate_self_attention(query, keys, values)
    attention_probabilities = softmax(attention_weights)
    predicted_index = np.random.choice(range(len(words)), p=attention_probabilities)
    predicted_word = index_to_word[predicted_index]
    return predicted_word, attention_probabilities
```

## Full Source Code

```
import numpy as np

def softmax(x):
  """Computes the softmax function for a numpy array.

  Args:
    x: A numpy array of shape (N,).

  Returns:
    A numpy array of shape (N,) containing the softmax probabilities.
  """

  # Prevent numerical overflow by subtracting the maximum value from all elements
  x -= np.max(x)

  # Compute the exponentials
  exp_x = np.exp(x)

  # Normalize the exponentials to sum to 1
  softmax_x = exp_x / np.sum(exp_x)

  return softmax_x

def create_word_representations(sentences):
    word_to_index = {}
    index_to_word = {}
    word_embeddings = []

    for sentence in sentences:
        for word in sentence.split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word[len(index_to_word)] = word
                word_embeddings.append(np.random.rand(3))  # Random embeddings

    return np.array(word_embeddings), word_to_index, index_to_word

def calculate_self_attention(query, keys, values):
    # Calculate scores for each word in the context
    scores = np.dot(query, keys.T) / np.sqrt(keys.shape[1])

    # Handle unique words (1D arrays)
    attention_weights = np.empty_like(scores)
    for i in range(len(scores)):
        if len(keys[i].shape) == 1:  # Check if 1D array
            attention_weights[i] = np.exp(scores[i])  # No need to sum for unique words
        else:
            attention_weights[i] = np.exp(scores[i]) / np.sum(np.exp(scores[i]), axis=1, keepdims=True)

    return attention_weights

def predict_next_word_with_self_attention(current_word, context_window, words, word_embeddings, word_to_index, index_to_word):
    context_embeddings = word_embeddings[[word_to_index[word] for word in context_window]]
    query = np.mean(context_embeddings, axis=0)  # Average context embeddings
    keys = values = np.array([word_embeddings[word_to_index[word]] for word in words])
    attention_weights = calculate_self_attention(query, keys, values)
    # Use softmax for probability distribution
    attention_probabilities = softmax(attention_weights)
    predicted_index = np.random.choice(range(len(words)), p=attention_probabilities)
    predicted_word = index_to_word[predicted_index]
    return predicted_word, attention_probabilities

if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog",
        "She sells seashells by the seashore",
        "I love to eat pizza with extra cheese",
    ]

    word_embeddings, word_to_index, index_to_word = create_word_representations(sentences)
    current_word = "fox"
    context_window_size = 2  # Considering two words before the current word

    for sentence in sentences:
        words = sentence.split()
        predicted_word, attention_probabilities = predict_next_word_with_self_attention(current_word, words[-(context_window_size+1):], words, word_embeddings, word_to_index, index_to_word)
        print(f"Given the word: {current_word}")
        print(f"Context: {' '.join(words[-(context_window_size+1):])}")  # Print context window
        print(f"Sentence: {sentence}")
        print("Attention Probabilities:")
        for word, prob in zip(words, attention_probabilities):
            print(f"\t{word}: {prob:.4f}")
        print(f"Predicted next word: {predicted_word}\n")
```

## Conclusion

The self-attention mechanism is a powerful tool in NLP, allowing models to focus on different parts of the input when producing an output. This code provides a simple example of how to implement this mechanism in Python, and can serve as a starting point for more complex NLP tasks.
