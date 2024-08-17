![demo](https://arxiv.org/html/2310.01061v2/x2.png)

# **Demystifying GraphRAG: Contextual Reasoning for AI**

## **Introduction**

Have you ever felt like your AI just isn't getting the whole picture? You feed it data, but the outputs still seem...off. Maybe it's missing key context, or struggling to grasp the relationships between different pieces of information.

This is where **context augmentation** comes in. It's the secret sauce for taking large language models (LLMs) to the next level, by providing them with richer and more relevant information to work with.

In this blog post, we'll delve into the world of context augmentation, exploring a technique called **GraphRAG**. We'll break down how it works, its potential benefits, and the challenges it faces. We'll even see a practical example using MongoDB!

## The Unifying Goal: Context Augmentation

GraphRAG, Hybrid RAG, Hybrid GraphRAG, Agentic RAG, etc. At the core of these techniques, we find a singular objective: enriching the context available to LLMs. 
The fundamental aim is to provide models with more comprehensive and relevant information in order to improve output quality. 
This pursuit is a dynamic one, with new strategies and techniques continually emerging. 

While the methods may vary, the ultimate goal remains consistent: optimizing context augmentation for enhanced LLM performance.

## What is GraphRAG?

**GraphRAG** is a method that combines knowledge graphs with Retrieval-Augmented Generation (RAG) to enhance language model capabilities. It aims to provide richer, more structured context for LLMs by leveraging the interconnected nature of information within a knowledge graph.

### The Core of GraphRAG

A key characteristic of GraphRAG is its heavy reliance on LLMs for two critical tasks:

1. **Automatic Knowledge Graph Construction:** GraphRAG often employs LLMs to extract entities and relationships from textual data, automatically building a knowledge graph. This approach, while promising, introduces challenges related to LLM limitations such as bias, hallucinations, and difficulty in capturing complex relationships.
2. **Query Understanding and Graph Traversal:** To effectively utilize the knowledge graph, GraphRAG relies on the LLM to understand user queries, identify relevant entities, and navigate the graph structure. This process can be error-prone, especially when dealing with ambiguous or complex queries.

While GraphRAG offers potential benefits in terms of providing structured context, its effectiveness is contingent upon the accuracy and completeness of the automatically constructed knowledge graph, as well as the LLM's ability to accurately interpret queries and traverse the graph.

* Knowledge graphs can be used in two ways within RAG:
    * As a data store to retrieve information from.
    * As a data store of the semantic structure to retrieve vector chunks from.
* Scenarios where knowledge graphs are most relevant involve:
    * Conceptual aggregation: Combining information from multiple sources.
    * Conceptual alignment: Aligning information within an enterprise workflow.
    * Hierarchical retrieval: Deterministic retrieval through categories.
    * Hierarchical recommendation: Improved recommendation systems.
    * Personalization/Memory: Tailoring LLM responses based on user interactions.

## GraphRAG Pipeline

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

The prospect of constructing knowledge graphs directly from raw data using LLMs is undeniably alluring. However, this endeavor is fraught with significant challenges.

### Controlling LLM Output

A critical aspect often overlooked is the challenge of controlling LLM output to align with specific requirements. While LLMs can generate text, ensuring the output adheres to a desired structure or format is non-trivial. For instance, extracting relationships might require the LLM to produce structured outputs like tuples or dictionaries.

To address this, strategies such as:

* **Function Calling:** Integrating functions within the LLM's capabilities can help structure the output, ensuring it meets specific requirements.
* **Prompt Engineering:** Carefully crafting prompts can guide the LLM towards producing the desired output format.
* **Reinforcement Learning from Human Feedback (RLHF):** Training the LLM to follow specific guidelines through human feedback can improve output control.

## Python Example (using MongoDB)
```
from pymongo import MongoClient
from openai import AzureOpenAI
import json 

# Replace with your actual values
AZURE_OPENAI_ENDPOINT = "https://.openai.azure.com"
AZURE_OPENAI_API_KEY = "" 
deployment_name = "gpt-4-32k"  # The name of your model deployment
az_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY)

# Connect to MongoDB
client = MongoClient("")
db = client["DEMO"]  # Replace "your_database" with your actual database name

# Define people collection (replace with your collection name)
people_collection = db["__people"]

# Delete all documents from the collection (for a clean demo)
people_collection.delete_many({})
print("Deleted existing documents from people collection.")

# Sample data
sample_data = [
    {
        "name": "Alice",
        "referrals": ["Bob", "Charlie"],  # Alice referred Bob and Charlie
        "actions": [
            {"action": "made a purchase", "outcome": "received product"},
            {"action": "wrote a review", "outcome": "positive feedback"},
            {"action": "referred a friend", "outcome": "earned referral bonus"},
            {"action": "referred a friend", "outcome": "earned referral bonus"}
        ],
        "customer_segment": "loyal"
    },
    {
        "name": "Bob",
        "referrals": ["David"],  # Bob referred David
        "actions": [
            {"action": "made a purchase", "outcome": "received product"},
            {"action": "wrote a review", "outcome": "positive feedback"},
            {"action": "referred a friend", "outcome": "earned referral bonus"}
        ],
        "customer_segment": "new"
    },
    {
        "name": "Charlie",
        "referrals": ["Eve"],  # Charlie referred Eve
        "actions": [
            {"action": "made a purchase", "outcome": "received product"},
            {"action": "referred a friend", "outcome": "earned referral bonus"}
        ],
        "customer_segment": "new"
    },
    {
        "name": "David",
        "referrals": [],  # David didn't refer anyone
        "actions": [
            {"action": "made a purchase", "outcome": "received product"},
            {"action": "recommended product", "outcome": "positive feedback"}
        ],
        "customer_segment": "new"
    },
    {
        "name": "Eve",
        "referrals": [],  # Eve didn't refer anyone
        "actions": [
            {"action": "made a purchase", "outcome": "received product"},
            {"action": "wrote a review", "outcome": "positive feedback"}
        ],
        "customer_segment": "new"
    }
]

# Insert sample data into the collection
people_collection.insert_many(sample_data)
print("Inserted sample data into people collection.")

raw_text = """
    Alice, a tech enthusiast, found a product she liked and shared it with her friends, Bob and Charlie. Alice made a purchase and was so satisfied that she wrote a positive review online. A few weeks later, Alice bought another unit of the product as a gift for a friend.

    Bob, intrigued by Alice's recommendation, decided to make a purchase. He was so satisfied that he also wrote a positive review and shared it with his friend, David. A month later, Bob bought the product again for his brother. 

    David, initially skeptical, read Bob's review and decided to give it a try. He made a purchase and was so impressed that he recommended the product to his colleagues at work. After using the product for a while, David decided to buy another one as a backup.

    Charlie, after careful consideration and reading Alice's review, also decided to buy the product. He was so impressed that he recommended it to his friend, Frank. Frank, inspired by Charlie, decided to buy the product as well and left a positive review online. A few days later, Frank bought another unit of the product for his girlfriend.
"""

print("NEW TEXT: ", raw_text)
print("\nExtracting customer relationships...")
# Knowledge Graph Construction: Create a comprehensive knowledge graph from various data sources (text, structured data, etc.).
response = az_client.chat.completions.create(
      model=deployment_name,
      messages=[
          {"role": "system", "content": "You are a helpful assistant that extracts insights from text."},
          {"role": "system", "content": """
    [response format]
        [{
            "name": "A",
            "referrals": ["B", "C"],
            "actions": [
                {"action": "made a purchase", "outcome": "received product"},
                {"action": "referred a friend", "outcome": "earned referral bonus"},
                {"action": "referred a friend", "outcome": "earned referral bonus"}
            ],
            "customer_segment": "new|loyal"
        },...]   
    [end response format]
    ** IMPORTANT! MUST BE VALID JSON! **
"""},
          {"role": "user", "content": raw_text}
    ],
  )

# Rename 'relationships' to 'customer_relationships'
customer_relationships = json.loads(response.choices[0].message.content.strip())

print("\nExtracted customer relationships:")
print("====================================================")
for relationship in customer_relationships:
    print(f"{relationship}")

# Insert 'customer_relationships' into the collection
for relationship in customer_relationships:
    people_collection.update_one(
        {"name": relationship["name"]},  # filter
        {"$set": relationship},  # update
        upsert=True  # options
    )
print("Upserted customer relationships into people collection.")

# Query Understanding: Process and understand user queries to identify relevant entities and relations.
USER_QUESTION = "Question: Who did Alice refer (directly or indirectly) and what actions did they take? \n"
PRE_PROCESS_PROMPT = """
    What is the name of the person being asked about? Respond in JSON format.
    [response criteria]
    {
        "name": "Bob"
    }
    [end response criteria]
    ** IMPORTANT! MUST BE VALID JSON! **
"""
response = az_client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that extracts the name of the person being asked about."},
        {"role": "user", "content": USER_QUESTION+PRE_PROCESS_PROMPT}
    ],
)
# Print the answer
print(response.choices[0].message.content.strip())
filter2use = json.loads(response.choices[0].message.content.strip())


# Graph Traversal: Explore the knowledge graph based on the query, retrieving relevant information.
# Define the aggregation pipeline
pipeline = [
    {"$match": filter2use},
    {"$graphLookup": {
        "from": "__people",
        "startWith": "$referrals",
        "connectFromField": "referrals",
        "connectToField": "name",
        "depthField": "depth",
        "as": "referral_network"
    }},
    {"$unwind": "$referral_network"},
    {"$unwind": "$referral_network.actions"},
    {"$group": {
        "_id": {"name": "$referral_network.name", "action": "$referral_network.actions.action"},
        "count": {"$sum": 1},
        "outcome": {"$first": "$referral_network.actions.outcome"},
    }},
    {"$project": {"_id": 0, "action": "$_id.action", "count": 1, "outcome": 1, "name": "$_id.name"}}
]

# Execute the aggregation pipeline
result = people_collection.aggregate(pipeline)

# Print the question being asked
print("Question: Who did Alice refer (directly or indirectly) and what actions did they take?")
# Print the answer
print("Answer:")
first_answer = ""
for doc in list(result):
    first_answer += (f"- {doc['name']} performed the action '{doc['action']}' {doc['count']} time(s), resulting in the outcome: '{doc['outcome']}'\n")
print(first_answer)
print("====================================================")

# Contextual Enrichment: Combine retrieved graph information with original text for enhanced context.
# Construct the prompt
prompt = "Alice referred the following people (directly or indirectly): \n" + first_answer
prompt += "What could be their potential next actions based on their past actions and customer segment?"
print("PROMPT:")
print(prompt)
# Send the prompt to the az_client
# Response Generation: Utilize a language model to generate a comprehensive and informative response based on the enriched context.
response = az_client.chat.completions.create(
    model=deployment_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that predicts potential next actions based on past actions and customer segment."},
        {"role": "user", "content": prompt}
    ],
)
# Print the answer
print("Answer:")
print(response.choices[0].message.content.strip())
```
 
## Conclusion

The knowledge graph stands as a foundational component in the evolution of artificial intelligence, providing a structured framework for representing and connecting information. By serving as a comprehensive repository of entities, attributes, and relationships, the knowledge graph empowers AI systems to reason, learn, and adapt in ways previously unimaginable. 

By integrating graph knowledge we can build AI systems that can dynamically interact with external tools and databases, expanding their problem-solving abilities. Moreover, a shared knowledge graph facilitates collaboration among specialized AI agents, creating intelligent systems capable of addressing complex challenges from multiple perspectives. As knowledge graph technologies continue to mature, we can anticipate the emergence of AI systems that possess an increasingly sophisticated understanding of the world, enabling them to make informed decisions and take actions with human-like nuance. 


