## The Unifying Goal: Context Augmentation

At the core of these techniques, we find a singular objective: enriching the context available to LLMs. 
The fundamental aim is to provide models with more comprehensive and relevant information in order to improve output quality. 
This pursuit is a dynamic one, with new strategies and techniques continually emerging. 

While the methods may vary, the ultimate goal remains consistent: optimizing context augmentation for enhanced LLM performance.

## GraphRAG Pipeline

* Knowledge graphs can be used in two ways within RAG:
    * As a data store to retrieve information from.
    * As a data store of the semantic structure to retrieve vector chunks from.
* Scenarios where knowledge graphs are most relevant involve:
    * Conceptual aggregation: Combining information from multiple sources.
    * Conceptual alignment: Aligning information within an enterprise workflow.
    * Hierarchical retrieval: Deterministic retrieval through categories.
    * Hierarchical recommendation: Improved recommendation systems.
    * Personalization/Memory: Tailoring LLM responses based on user interactions.
    
**Core Concept:** Leverage a knowledge graph to enhance Retrieval-Augmented Generation (RAG).

**Pipeline:**

1. **Knowledge Graph Construction:** Create a comprehensive knowledge graph from various data sources (text, structured data, etc.).
2. **Query Understanding:** Process and understand user queries to identify relevant entities and relations.
3. **Graph Traversal:** Explore the knowledge graph based on the query, retrieving relevant information.
4. **Contextual Enrichment:** Combine retrieved graph information with original text for enhanced context.
5. **Response Generation:** Utilize a language model to generate a comprehensive and informative response based on the enriched context.

**Key Components:**

* **Knowledge Graph:** A structured representation of entities and their relationships.
* **Query Understanding Module:** Interprets user queries and extracts key information.
* **Graph Traversal Algorithm:** Efficiently navigates the knowledge graph.
* **Contextual Fusion Module:** Combines graph information with textual context.
* **Language Model:** Generates human-like text based on provided information.

By integrating a knowledge graph into the RAG process, GraphRAG aims to improve response accuracy, coherence, and factuality.

## The Challenge of Automatic Relationship Extraction

While the concept of automatically building knowledge graphs from raw data is appealing, the reality has many challenges.

* **LLM Limitations:**
  * **Bias:** LLMs are trained on massive datasets that can contain biases, leading to skewed relationship extraction.
  * **Hallucinations:** They can invent relationships that don't exist, compromising data integrity.
  * **Limited Understanding:** Deep understanding of complex relationships, especially domain-specific ones, remains elusive.
* **Data Quality:**
  * **Noise:** Impurities in data can lead to incorrect relationship extraction.
  * **Ambiguity:** Textual data can be ambiguous, making accurate interpretation difficult.
* **Domain Specificity:**
  * **Unique Relationships:** Industries often have specific terminology and relationship types not captured in general language models.
## The Challenge of Automatic Relationship Extraction

The prospect of constructing knowledge graphs directly from raw data using LLMs is undeniably alluring. However, this endeavor is fraught with significant challenges.

### LLM Limitations

* **Bias:** LLMs are trained on massive datasets that inevitably contain biases. These biases can be subtly or overtly reflected in the extracted relationships, leading to skewed and potentially harmful knowledge graphs.
* **Hallucinations:** LLMs are prone to fabricating information, a phenomenon known as hallucination. In the context of relationship extraction, this can result in the invention of nonexistent connections, compromising the integrity of the knowledge graph.
* **Limited Understanding:** While LLMs have shown remarkable capabilities, their comprehension of complex relationships, especially those specific to particular domains, remains limited. This hinders their ability to accurately capture nuanced connections between entities.

### Controlling LLM Output

A critical aspect often overlooked is the challenge of controlling LLM output to align with specific requirements. While LLMs can generate text, ensuring the output adheres to a desired structure or format is non-trivial. For instance, extracting relationships might require the LLM to produce structured outputs like tuples or dictionaries.

To address this, strategies such as:

* **Function Calling:** Integrating functions within the LLM's capabilities can help structure the output, ensuring it meets specific requirements.
* **Prompt Engineering:** Carefully crafting prompts can guide the LLM towards producing the desired output format.
* **Reinforcement Learning from Human Feedback (RLHF):** Training the LLM to follow specific guidelines through human feedback can improve output control.

## Python Example

```
import openai

def extract_entities_with_azure_openai(text):
  """
  Extracts entities from the text using Azure OpenAI's language model.

  Args:
      text (str): The text to extract entities from.

  Returns:
      list: A list of extracted entities.
  """
  response = openai.Completion.create(
      engine="text-davinci-003",  # Replace with your chosen model
      prompt=f"Extract named entities from the following text: {text}",
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5
  )
  entities = response.choices[0].text.strip().split("\n")
  return entities

def extract_relationships_with_azure_openai(text, entities):
  """
  Extracts relationships between entities using Azure OpenAI's language model.

  This is a simplified approach using dependency parsing prompts.

  Args:
      text (str): The text to extract relationships from.
      entities (list): A list of extracted entities.

  Returns:
      list: A list of tuples representing relationships (start_entity, relationship, end_entity).
  """
  relationships = []
  for entity1 in entities:
    for entity2 in entities:
      if entity1 != entity2:
        prompt = f"What is the relationship between {entity1} and {entity2} in the sentence: {text}?"
        response = openai.Completion.create(
          engine="text-davinci-003",  # Replace with your chosen model
          prompt=prompt,
          max_tokens=256,
          n=1,
          stop=None,
          temperature=0.5
        )
        relationship = response.choices[0].text.strip()
        relationships.append((entity1, relationship, entity2))
  return relationships

def create_graph(entities, relationships):
  """
  Creates an in-memory graph representation using a dictionary.

  Args:
      entities (list): A list of extracted entities.
      relationships (list): A list of tuples representing relationships (start_entity, relationship, end_entity).

  Returns:
      dict: A dictionary representing the graph structure.
  """
  graph = {}  # Initialize an empty dictionary to represent the graph

  # Add nodes (entities) to the graph
  for entity in entities:
    graph[entity] = {}  # Each entity key points to an empty dictionary for its connections

  # Add relationships (edges) to the graph
  for start, rel, end in relationships:
    graph[start][end] = rel  # Add the relationship as a key-value pair on the starting entity

  return graph

# Example usage
text = "Apple is a technology company founded by Steve Jobs. It develops smartphones like iPhone."
entities = extract_entities_with_azure_openai(text)
relationships = extract_relationships_with_azure_openai(text, entities)

graph = create_graph(entities, relationships)

# Accessing information in the graph
print(graph["Apple"])  # Prints connections for the "Apple" entity

# Iterating through relationships
for start_entity, connections in graph.items():
  for end_entity, relationship in connections.items():
    print(f"{start_entity} -> {relationship} -> {end_entity}")
```

**The Role of Human Expertise**

To mitigate these challenges, human expertise is essential.

* **Context Augmentation:** Providing domain-specific context to the LLM can improve accuracy.
* **Relationship Definition:** Explicitly defining desired relationships helps guide the model.
* **Quality Control:** Human evaluation and validation are crucial for ensuring data accuracy.

**Hybrid Approach**

* **Human-in-the-Loop:** Humans can provide initial structure and corrections.
* **AI-Assisted Refinement:** LLMs can suggest potential relationships based on patterns.
* **Iterative Improvement:** Continuous refinement through feedback loops.

 
**Conclusion**

While the promise of automated knowledge graph creation is enticing, a human-centric approach is essential for building high-quality, trustworthy knowledge bases. 
By effectively combining human intelligence with AI capabilities, we can unlock the full potential of this technology.

**The Future of GraphRAG:**

* **Small graphs and multi-agent systems:** Focus on creating smaller, task-specific knowledge graphs for complex information retrieval processes.
* **Long-term strategic moat:** Companies can leverage knowledge graphs as a core asset to exploit their domain-specific data and create a competitive advantage.
* **GraphRAG for complex reasoning tasks:** Developing techniques for handling complex queries and reasoning chains.

** What about MongoDB? **

```
import openai
import pymongo

# Replace with your Azure OpenAI API key and endpoint
openai.api_key = "YOUR_AZURE_OPENAI_API_KEY"
openai.api_base = "YOUR_AZURE_OPENAI_ENDPOINT"

# Replace with your MongoDB connection details
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["your_database_name"]  # Replace with your database name
graph_collection = db["knowledge_graph"]  # Replace with your collection name

def extract_entities_with_azure_openai(text):
  """
  Extracts entities from the text using Azure OpenAI's language model.

  Args:
      text (str): The text to extract entities from.

  Returns:
      list: A list of extracted entities.
  """
  response = openai.Completion.create(
      engine="text-davinci-003",  # Replace with your chosen model
      prompt=f"Extract named entities from the following text: {text}",
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5
  )
  entities = response.choices[0].text.strip().split("\n")
  return entities

def extract_relationships_with_azure_openai(text, entities):
  """
  Extracts relationships between entities using Azure OpenAI's language model.

  This is a simplified approach using dependency parsing prompts.

  Args:
      text (str): The text to extract relationships from.
      entities (list): A list of extracted entities.

  Returns:
      list: A list of dictionaries representing relationships (start_entity, relationship, end_entity).
  """
  relationships = []
  for entity1 in entities:
    for entity2 in entities:
      if entity1 != entity2:
        prompt = f"What is the relationship between {entity1} and {entity2} in the sentence: {text}?"
        response = openai.Completion.create(
          engine="text-davinci-003",  # Replace with your chosen model
          prompt=prompt,
          max_tokens=256,
          n=1,
          stop=None,
          temperature=0.5
        )
        relationship = response.choices[0].text.strip()
        relationships.append({"start_entity": entity1, "relationship": relationship, "end_entity": entity2})
  return relationships

def create_knowledge_graph(entities, relationships):
  """
  Creates a knowledge graph and stores it in MongoDB using $graphLookup pipeline.

  Args:
      entities (list): A list of extracted entities.
      relationships (list): A list of dictionaries representing relationships (start_entity, relationship, end_entity).
  """
  graph_data = []
  for entity in entities:
    graph_data.append({"entity": entity})

  # Add relationships to each entity data
  for entity_data in graph_data:
    entity_name = entity_data["entity"]
    connected_entities = [rel["end_entity"] for rel in relationships if rel["start_entity"] == entity_name]
    entity_data["connected_entities"] = connected_entities

  # Upsert (update or insert) the graph data in MongoDB
  graph_collection.bulk_write([pymongo.InsertOne(item) for item in graph_data], ordered=False)

def lookup_information_in_graph(start_entity, depth=1):
  """
  Looks up information in the knowledge graph using $graphLookup aggregation.

  Args:
      start_entity (str): The starting entity to traverse the graph from.
      depth (int, optional): The maximum depth of traversal (defaults to 1).

  Returns:
      list: A list of dictionaries representing connected entities at the specified depth.
  """
  pipeline = [
      {
          "$match": {"entity": start_entity}
      },
      {
          "$lookup": {
              "from": "knowledge_graph",
              "localField": "connected_entities",
              "foreignField": "entity",
              "as": "connected_entities",
              "depth": depth
          }
      },
      {
          "$unwind": "$connected_entities"
      },
      {
          "$project": {"_id": 0, "entity": 1, "connected_entities": 1}
      }
  ]
  results = list(graph_collection.aggregate(pipeline))

  # Remove duplicates
  unique_results = []
  seen_entities = set()
  for result in results:
    if result["entity"] not in seen_entities:
      unique_results.append(result)
      seen_entities.add(result["entity"])

  return unique_results

def augment_prompt_with_graph_data(query, start_entity, graph_lookup_function):
  """
  Augments the prompt with information retrieved from the knowledge graph.

  Args:
      query (str): The original user query.
      start_entity (str): The starting entity for the graph lookup.
      graph_lookup_function: The function to perform the graph lookup.

  Returns:
      str: The augmented prompt.
  """
  graph_data = graph_lookup_function(start_entity)
  # Process graph_data to extract relevant information
  augmented_prompt = f"{query}\n\nAdditional context: {graph_data}"
  return augmented_prompt

def generate_response(prompt):
  """
  Generates a response using the provided prompt.

  Args:
      prompt (str): The augmented prompt.

  Returns:
      str: The generated response.
  """
  response = openai.Completion.create(
      engine="text-davinci-003",  # Replace with your chosen model
      prompt=prompt,
      max_tokens=1024,
      n=1,
      stop=None,
      temperature=0.5
  )
  return response.choices[0].text.strip()

# Example usage
text = "Apple is a technology company founded by Steve Jobs. It develops smartphones like iPhone."
entities = extract_entities_with_azure_openai(text)
relationships = extract_relationships_with_azure_openai(text, entities)

create_knowledge_graph(entities, relationships)

query = "Tell me about Apple"
start_entity = "Apple"
augmented_prompt = augment_prompt_with_graph_data(query, start_entity, lookup_information_in_graph)
response = generate_response(augmented_prompt)
print(response)
```
