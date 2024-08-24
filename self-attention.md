
# **We should pay more attention to attention**

It all started with: "Attention is all you need."

Imagine you're reading a long novel. You don't read every word with the same level of focus. Instead, you pay more attention to certain parts, like the plot twists or character developments. This is similar to how attention works in a language model.

Attention is a mechanism that allows a transformer to focus on different parts of its input sequence based on their relevance to the current output. This is achieved by assigning weights to each input element, with larger weights indicating greater importance. These weights are calculated using a similarity metric, such as the dot product, between the query vector and each key vector in the input sequence.

**Attention and Quality:**

* **Positive impact:** Attention allows LLMs to focus on the most relevant parts of the input sequence when generating a response. This leads to responses that are more coherent, relevant, and grammatically correct.
* **Negative impact:**  
    * **Focus on misleading information:**  If the input contains misleading or irrelevant keywords, the LLM's attention might be drawn to them, resulting in inaccurate or nonsensical responses.
    * **Missing key information:**  The LLM might overlook crucial information if the wording is different from what it's trained on. 

**The Illusion of "Intelligence"**

The term "intelligence" is often used to describe the capabilities of mathematical models like LLMs. However, it's important to note that this "intelligence" is heavily dependent on the quality and diversity of the training data. The model's "intelligence" can be biased, limited, and even misleading. For instance, an LLM trained on mostly news articles might struggle to understand sarcasm or humor. 

Moreover, the concept of "intelligence" in the context of automating unpredictable workflows is problematic for LLMs. Due to the way 'attention' works, LLMs can be manipulated, leading to potentially undesirable outcomes.

## Introduction

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

This code implements a basic model that uses **self-attention** to predict the next word in a sentence. It doesn't have a specific name for the entire model itself, but the core functionality relies on the self-attention mechanism.

Here's a breakdown:

* **Self-Attention**: This is the key concept used in the `calculate_self_attention` function. It allows the model to focus on relevant parts of the input sequence (the sentence) when predicting the next word.

* **Word Embeddings**: The model uses randomly generated embeddings to represent each word. These embeddings are then projected into query, key, and value vectors which are used for calculating the attention weights.

**Overall, the model can be considered a simple recurrent neural network (RNN) with a self-attention mechanism for next word prediction.** It demonstrates the core idea of self-attention but lacks the complexity of more advanced models like Transformers, which utilize this mechanism extensively.

## The Impact of Randomly Generated Embeddings

**Randomly generated embeddings** serve as a starting point for the model to learn meaningful representations of words. They are essentially arbitrary numerical vectors assigned to each word.

**Why use random embeddings?**

1. **Initialization**: Randomly generated embeddings provide a non-zero initial state for the model. This allows the model to start learning and adjusting the embeddings based on the input data.
2. **Breaking Symmetry**: Random initialization helps to avoid symmetry in the model's parameters, which can hinder learning.
3. **Exploration**: Randomness can encourage the model to explore different parts of the solution space, potentially leading to better performance.

**How do they evolve?**

As the model trains on more data, the embeddings are updated through backpropagation. The model learns to adjust the embeddings so that words with similar meanings have similar representations. For example, words like "cat" and "dog" might end up having similar embeddings because they are often used in similar contexts.

**Limitations of Random Initialization:**

* **Local Minima**: Random initialization can sometimes lead the model to get stuck in local minima, preventing it from reaching the global optimum.
* **Slower Convergence**: Random initialization can sometimes require more training epochs to converge to a good solution.

In summary, while randomly generated embeddings may seem arbitrary at first, they play a crucial role in initializing the model and allowing it to learn meaningful representations of words.

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

The attention weights show how much importance the model assigns to each word in the context when predicting the next word. Higher weights indicate greater relevance.

- The: 2.1307
- quick: 2.5428
- brown: 1.9087
- fox: 2.6365
- jumps: 2.2119
- over: 1.2500
- the: 2.1166
- lazy: 2.5802
- dog: 1.5677

As you can see, the words "quick," "fox," and "lazy" have the highest weights, suggesting they are the most important for predicting the next word.

## Predicting the Next Word with Self-Attention

The `predict_next_word_with_self_attention` function uses the self-attention mechanism to predict the next word in a sentence. It first calculates the context embeddings by averaging the embeddings of the words in the context window. It then calculates the attention weights and applies the softmax function to get a probability distribution over the words. The next word is predicted by sampling from this distribution.

```python
def predict_next_word_with_self_attention(current_word, context_window, words, word_embeddings, word_to_index, index_to_word):
    context_embeddings = word_embeddings[[word_to_index[word] for word in context_window]]
    query = np.mean(context_embeddings, axis=0)  # Average context embeddings
    keys = values = np.array([word_embeddings[word_to_index[word]] for word in words])
    attention_weights = calculate_self_attention(query, keys, values)
    attention_probabilities = softmax(attention_weights)
    predicted_index = np.argmax(attention_probabilities)  # Select the word with the highest probability
    predicted_word = index_to_word[predicted_index]
    return predicted_word, attention_probabilities
```

## Breaking Down the Prediction Process

This code is implementing a simple version of the self-attention mechanism, which is a key component in Transformer models used in natural language processing. The self-attention mechanism allows the model to weigh the importance of words in a sentence when predicting the next word.

Here's a breakdown of the code:

1. `create_word_representations(sentences)`: This function takes a list of sentences as input and creates a word-to-index and index-to-word dictionary, and a list of word embeddings. Each unique word in the sentences is assigned a unique index and a random 3-dimensional vector as its embedding.

2. `calculate_self_attention(query, key, value)`: This function calculates the self-attention weights and the output vector. The attention weights are calculated by taking the dot product of the query and key, scaling it, and applying the softmax function. The output vector is the weighted sum of the value vectors, where the weights are the attention weights.

3. `predict_next_word_with_self_attention(current_word, words, word_embeddings, word_to_index, index_to_word)`: This function predicts the next word given the current word and a list of words (context). It first retrieves the embeddings of the current word and the context words, then calculates the self-attention weights and output vector. The predicted next word is the word whose embedding is closest to the output vector in terms of Euclidean distance.

4. The main part of the code creates word representations for a list of sentences, then for each sentence, it predicts the next word given the current word "fox" and prints the attention weights for each word in the sentence and the predicted next word.

Please note that this is a simplified and not a practical implementation of self-attention. In a real-world scenario, the word embeddings would be learned from data rather than randomly assigned, and the attention mechanism would consider all words in the context, not just the current word.

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
        current_word_index = words.index(current_word)
        context_window = words[max(0, current_word_index - context_window_size):current_word_index]
        predicted_word, attention_probabilities = predict_next_word_with_self_attention(current_word, context_window, words, word_embeddings, word_to_index, index_to_word)
        print(f"\nGiven the word: {current_word}")
        print(f"Context: {' '.join(context_window)}")  # Print context window
        print(f"Sentence: {sentence}")
        print("Attention Probabilities:")
        for word, prob in zip(words, attention_probabilities):
            print(f"\t{word}: {prob:.4f}")
        print(f"Predicted next word: {predicted_word}\n")
```

This code provides a basic model that uses self-attention to predict the next word in a sentence. It demonstrates the core idea of self-attention but lacks the complexity of more advanced models like Transformers, which utilize this mechanism extensively.

## **Additional Considerations:**

- The quality of the word embeddings used can significantly impact the model's performance.
- The size of the vocabulary and the complexity of the language can also affect the model's accuracy.

**Intelligence is a Product of Training**

The "intelligence" of an LLM is directly tied to the quality and diversity of its training data. Here's how:

* **Data Bias:** If the training data is biased, the LLM will also be biased in its outputs. For example, an LLM trained on mostly news articles might struggle to understand sarcasm or humor. 
* **Data Limitedness:** The real world is vast and complex. LLMs can only process what they've been trained on. Limited data can lead to incomplete understanding and difficulty handling unexpected situations.
* **Training Objectives:** Ultimately, LLMs are optimized for the tasks they are trained on. An LLM trained for text summarization may not excel at creative writing tasks, even if the data is vast.

## Conclusion

**The Power of Attention, with Caveats**

In this exploration, we've delved into the concept of attention in large language models (LLMs). Attention acts as a spotlight, allowing LLMs to focus on crucial parts of the input sequence, leading to more coherent and relevant outputs. However, it's essential to recognize the limitations of attention and understand that it doesn't equate to true comprehension. 

**Beyond the Spotlight: The Challenges of LLM Intelligence**

The "intelligence" of an LLM heavily depends on the quality and variety of its training data. Biases, limitations in the data itself, and narrow training objectives can all hinder a model's ability to represent the real world's complexities. Just like a student highlighting doesn't guarantee comprehension, attention in LLMs doesn't guarantee true understanding. 

**Looking Forward: Responsible Development and Usage**

As LLMs continue to evolve, it's crucial for developers to prioritize high-quality, diverse training data to mitigate bias and limitations. Additionally, we, as users, must be aware of these limitations and approach LLM outputs with a critical eye. By understanding both the power and limitations of attention, we can foster responsible development and usage of LLMs, ensuring they serve as valuable tools, not replacements for human intelligence.  


