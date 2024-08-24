**We should pay more attention to attention**

It all started with: "Attention is all you need."

Imagine you're reading a long novel. You don't read every word with the same level of focus. Instead, you pay more attention to certain parts, like the plot twists or character developments. This is similar to how attention works in a language model.

Attention is a mechanism that allows a transformer to focus on different parts of its input sequence based on their relevance to the current output. This is achieved by assigning weights to each input element, with larger weights indicating greater importance. These weights are calculated using a similarity metric, such as the dot product, between the query vector and each key vector in the input sequence.

**Attention and Quality:**

* **Positive impact:** Attention allows LLMs to focus on the most relevant parts of the input sequence when generating a response. This leads to responses that are more coherent, relevant, and grammatically correct.
* **Negative impact:**  
    * **Focus on misleading information:**  If the input contains misleading or irrelevant keywords, the LLM's attention might be drawn to them, resulting in inaccurate or nonsensical responses.
    * **Missing key information:**  The LLM might overlook crucial information if the wording is different from what it's trained on. 

**The flexibility is the vulnerability:**

The LLM's reliance on attention makes it susceptible to manipulation.  By crafting inputs with specific words or phrases that grab the LLM's attention, malicious actors can steer the model towards generating biased, offensive, or nonsensical content.

**How Attention Works in Transformers**

Imagine the LLM as a spotlight. Attention controls where the light shines. Ideally, it illuminates the important parts of the scene.  However, if there's a shiny object in the corner, the spotlight might get drawn to that, neglecting the main subject.  In the same way, misleading words in a chunk can divert the LLM's attention and lead to poor response quality.

In a transformer, each input token is transformed into three vectors: a query, a key, and a value. The query vector represents the current position or token we're focusing on, while the key vectors represent all other positions or tokens in the sequence. The value vectors represent the actual information associated with each token.

The dot product is computed between the query vector and each key vector. This is essentially a measure of how related each input token is to the current position. The dot products are then passed through a softmax function to obtain attention weights, which sum up to 1. These weights represent the probability distribution over all input tokens for the current position.

## LLMs: When the Magic Starts to Fade

Large Language Models (LLMs) have undeniably revolutionized the way we interact with technology. Their ability to generate human-quality text, translate languages, write different kinds of creative content, and answer your questions in an informative way is nothing short of astonishing. But what happens when these models encounter the unexpected?

To understand the limitations of LLMs, let's revisit our oversimplified model trained on three sentences:

1. The quick brown fox jumps over the lazy dog.
2. She sells seashells by the seashore.
3. I love to eat pizza.
4. The cat chased the mouse. 
5. The sun is shining brightly.
6. I am feeling happy today. 

While still a simplified representation, this dataset allows us to explore some complexities.

### The Illusion Shatters

Let's introduce an ambiguous sentence: "I saw a man in the park with a telescope." Our expanded LLM, while more sophisticated than the initial one, still faces challenges. It might generate outputs like:

* **Incorrect interpretation:** "The man chased the mouse with a telescope in the park."
* **Reliance on familiar patterns:** "The sun is shining brightly on the man with a telescope."
* **Incoherent output:** A nonsensical combination of words.

These examples highlight the limitations of LLMs in handling ambiguity and context. While they can generate impressive text based on patterns learned from vast amounts of data, they struggle to understand the nuances of human language.

## A Simple Language Model with Attention

In this blog post, we'll explore a basic implementation of a language model capable of predicting the next word in a sequence based on a given word. This model incorporates a simplified attention mechanism to focus on relevant parts of the input sequence.

**Building the Language Model**

1. **Data Preparation:**
   - We start with a collection of sentences that will serve as our training data.
   - These sentences are broken down into individual words, creating a vocabulary.

2. **Word Frequency Calculation:**
   - Each word in the vocabulary is counted to determine its frequency within the training data.
   - This information is stored in a dictionary, where the keys are the words and the values are their corresponding frequencies.

3. **Word Embeddings:**
   - Words are represented as numerical vectors called embeddings, which capture semantic and syntactic information about the words.

4. **Attention Mechanism:**
   - A simplified attention mechanism is implemented to assign weights to each word in the input sequence based on its relevance to the current prediction.
   - The attention weights are calculated using a dot product between the current word's embedding and the embeddings of the other words in the sequence.

5. **Prediction:**
   - The weighted sum of the word embeddings is calculated using the attention weights.
   - The closest word in the vocabulary to this weighted sum is predicted as the next word.

**Example Implementation**

```python
import random
import numpy as np

# Training sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "I love to eat pizza.",
    "The cat chased the mouse.",
    "The sun is shining brightly.",
    "I am feeling happy today.",
]

# Create word frequency dictionary
word_freq = {}
for sentence in sentences:
    words = sentence.split()
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

# Convert words to numerical representations (simplified example)
word_to_index = {word: i for i, word in enumerate(word_freq.keys())}
word_embeddings = np.random.rand(len(word_freq), 3)  # Replace with actual embeddings

def calculate_attention_weights(current_word, words):
    # A simple dot product-based attention mechanism
    attention_scores = np.dot(word_embeddings[word_to_index[current_word]], np.array([word_embeddings[word_to_index[word]] for word in words]).T)
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
    return attention_weights

def predict_next_word_with_attention(current_word, words):
    attention_weights = calculate_attention_weights(current_word, words)
    weighted_words = np.array([word_embeddings[word_to_index[word]] for word in words]) * attention_weights[:, None]
    weighted_sum = np.sum(weighted_words, axis=0)
    closest_index = np.argmin(np.linalg.norm(word_embeddings - weighted_sum, axis=1))
    predicted_word = list(word_to_index.keys())[closest_index]
    return predicted_word, attention_weights

# Example usage
for sentence in sentences:
    current_word = "to"
    predicted_word, attention_weights = predict_next_word_with_attention(current_word, sentence.split())

    print("Given the word:", current_word)
    print("Attention weights for:", sentence)
    for word, weight in zip(sentence.split(), attention_weights):
        print(f"{word}: {weight:.4f}")

    print("Predicted next word:", predicted_word)
    print("\n")
```

**Explanation**

- The attention weights represent the importance that the model assigns to each word in the input sequence when predicting the next word.
- Words with higher weights are considered more relevant to the prediction.
- The weighted sum of word embeddings captures the context of the current word.
- The prediction is made based on the similarity between the context vector and the word embeddings in the vocabulary.

**Note on Varying Attention Weights**

The attention weights will vary depending on the specific input sentence. This is because the model is learning to dynamically adjust its focus based on the context of the words in the sequence. By analyzing the attention weights for different sentences, you can gain insights into how the model is interpreting and processing language.

