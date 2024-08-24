# **We should pay more attention to attention**

It all started with: "Attention is all you need."

Imagine you're reading a long novel. You don't read every word with the same level of focus. Instead, you pay more attention to certain parts, like the plot twists or character developments. This is similar to how attention works in a language model.

Attention is a mechanism that allows a transformer to focus on different parts of its input sequence based on their relevance to the current output. This is achieved by assigning weights to each input element, with larger weights indicating greater importance. These weights are calculated using a similarity metric, such as the dot product, between the query vector and each key vector in the input sequence.

**Attention and Quality:**

* **Positive impact:** Attention allows LLMs to focus on the most relevant parts of the input sequence when generating a response. This leads to responses that are more coherent, relevant, and grammatically correct.
* **Negative impact:**  
    * **Focus on misleading information:**  If the input contains misleading or irrelevant keywords, the LLM's attention might be drawn to them, resulting in inaccurate or nonsensical responses.
    * **Missing key information:**  The LLM might overlook crucial information if the wording is different from what it's trained on. 

In this blog post, we will explore the concept of attention through a Python code snippet that uses the self-attention mechanism to predict the next word in a sentence.

```python
import numpy as np
```

We start by importing the numpy library, which provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## The Softmax Function

The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or large, the softmax function squashes it, which helps in mitigating the exploding and vanishing gradient problems.

```python
def softmax(x):
  x -= np.max(x)
  exp_x = np.exp(x)
  softmax_x = exp_x / np.sum(exp_x)
  return softmax_x
```

## Creating Word Representations

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

## Calculating Self-Attention

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

## Predicting the Next Word with Self-Attention

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

## Running the Model

Finally, we run the model on a set of sentences. For each sentence, the model predicts the next word given the current word "fox" and prints the attention weights for each word in the sentence and the predicted next word.

```python
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

This code provides a basic model that uses self-attention to predict the next word in a sentence. It demonstrates the core idea of self-attention but lacks the complexity of more advanced models like Transformers, which utilize this mechanism extensively.

# The Power of Attention in Language Models: A Python Code Walkthrough

In the world of Natural Language Processing (NLP), the concept of attention has emerged as a transformative force. This idea was first brought to the forefront in the influential paper "Attention is all you need," and has since become a fundamental component of many modern NLP models, including the Transformer model.

In this blog post, we will explore the concept of attention through a Python code snippet that uses the self-attention mechanism to predict the next word in a sentence.

```python
import numpy as np
```

We start by importing the numpy library, which provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## The Softmax Function

The softmax function is a function that turns a vector of K real values into a vector of K real values that sum to 1. The input values can be positive, negative, zero, or greater than one, but the softmax transforms them into values between 0 and 1, so that they can be interpreted as probabilities. If one of the inputs is small or large, the softmax function squashes it, which helps in mitigating the exploding and vanishing gradient problems.

```python
def softmax(x):
  x -= np.max(x)
  exp_x = np.exp(x)
  softmax_x = exp_x / np.sum(exp_x)
  return softmax_x
```

## Creating Word Representations

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

## Calculating Self-Attention

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

## Predicting the Next Word with Self-Attention

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

## Running the Model

Finally, we run the model on a set of sentences. For each sentence, the model predicts the next word given the current word "fox" and prints the attention weights for each word in the sentence and the predicted next word.

```python
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

This code provides a basic model that uses self-attention to predict the next word in a sentence. It demonstrates the core idea of self-attention but lacks the complexity of more advanced models like Transformers, which utilize this mechanism extensively.
