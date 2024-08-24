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


