
![](https://dist.neo4j.com/wp-content/uploads/20230615211415/1mkYvs8_TmzLhUUI1CShNfw.png)

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
* **Query Understanding:** Interprets user queries and extracts key information.
* **Graph Traversal Algorithm:** Efficiently navigates the knowledge graph.
* **Contextual Fusion Module:** Combines graph information with textual context.
* **Language Model:** Generates human-like text based on provided information.

By integrating a knowledge graph into the RAG process, GraphRAG aims to improve response accuracy, coherence, and factuality.

## The Challenge of Automatic Relationship Extraction

![](https://dist.neo4j.com/wp-content/uploads/20230615211423/1N-TVTbRffy_VQ0DPcx0JKg.png)

While the concept of automatically building knowledge graphs from raw data is appealing, the reality has many challenges.

* **LLM Limitations:**
  * **Lost in the Middle Problem:** The attention mechanism for focusing on important content in the middle of long inputs might not prioritize mid-sequence information effectively.
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
from enum import Enum
from typing import List, Dict, Tuple
from openai import AzureOpenAI
import json 
from pymongo import MongoClient

# Replace with your actual values
MDB_URI = ""
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = "" 
deployment_name = "gpt-4o-mini"  # The name of your model deployment
az_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY)

class Relationship(Enum):
    WORKED_AT = "worked at"
    FOUNDED = "founded"

documents = [
    "Steve Jobs founded Apple.",
    "Before Apple, Steve Jobs worked at Atari.",
    "Steve Wozniak and Steve Jobs founded Apple together.",
    "After leaving Apple, Steve Jobs founded NeXT.",
    "Steve Wozniak and Steve Jobs worked together at Apple.",
    "Bill Gates founded Microsoft.",
    "Microsoft and Apple were rivals in the early days of the personal computer market.",
    "Bill Gates worked at Microsoft for many years before stepping down as CEO.",
    "Elon Musk founded SpaceX.",
    "Before SpaceX, Elon Musk founded PayPal.",
    "Elon Musk also founded Tesla, a company that produces electric vehicles.",
    "Jeff Bezos founded Amazon.",
    "Amazon started as an online bookstore before expanding into other markets.",
    "Jeff Bezos also founded Blue Origin, a space exploration company.",
    "Blue Origin and SpaceX are competitors in the private space industry."
]

class Node:
    """Represents a node in the knowledge graph."""
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

class Edge:
    """Represents an edge in the knowledge graph."""
    def __init__(self, source_node: Node, target_node: Node, relation: str):
        self.source_node = source_node
        self.target_node = target_node
        self.relation = relation

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.source_node.name == other.source_node.name and self.target_node.name == other.target_node.name and self.relation == other.relation
        return False

    def __hash__(self):
        return hash((self.source_node.name, self.target_node.name, self.relation))

class KnowledgeGraph:
    """Creates a knowledge graph from a list of documents."""
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.nodes = {}
        self.edges = []
        # Connect to MongoDB
        self.client = MongoClient(MDB_URI)

    def store_in_mongodb(self, db_name: str, collection_name: str):
        # Connect to MongoDB
        db = self.client[db_name]
        collection = db[collection_name]
        collection.delete_many({})
        # Convert nodes and edges to a format suitable for MongoDB
        for name, node in self.nodes.items():
            node_data = {'_id': name, 'type': node.type, 'edges': []}
            for edge in self.edges:
                if edge.source_node.name == name:
                    node_data['edges'].append({'relation': edge.relation, 'target': edge.target_node.name})
                    print(edge.source_node.name, edge.relation, edge.target_node.name)
            collection.insert_one(node_data)
    
    def create_knowledge_graph(self):
        """Creates a knowledge graph from the list of documents."""
        for document in self.documents:
            relationships = []
            prompt = f"Identify relationships in the text: ```{str(document)}```\n"
            prompt += "Following relationships are possible: ```"
            prompt += ", ".join([rel.value for rel in Relationship])
            prompt += """```
    Format concise response as a JSON object with only two keys called "relationships", and "nodes". 
    The value of the "relationships" key should be a list of objects each with these fields (source, source_type, relation, target, target_type).
    IF NO RELATIONSHIP IS FOUND, RETURN EMPTY LIST.
    IF NO NODES ARE FOUND, RETURN EMPTY LIST.

    [response criteria]
    - JSON object: { "relationships": [], "nodes": [] }
    - each relationship should be of the format: { "source": "Alice", "source_type": "person", "target": "MongoDB", "relation": "worked at", "target_type": "company" }
    - each node should be of the format: { "name": "MongoDB", "type": "company" }
    [end response criteria]
    """
            try:
                response = az_client.chat.completions.create(
                            model=deployment_name,
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant that extracts the name of the person being asked about."},
                                {"role": "system", "content": "You specialize in identifying these relationships: " + ", ".join([rel.value for rel in Relationship])},
                                {"role": "user", "content": prompt},
                            ],
                            response_format={ "type": "json_object" }
                )
                completion = json.loads(response.choices[0].message.content.strip())
                
                # Parse the OpenAI response
                for r in completion["relationships"]:
                    relationships.append((r["source"],r["source_type"],r["target"], r["relation"],r["target_type"]))
                for n in completion["nodes"]:
                    self.nodes[n["name"]] = Node(n["name"],n["type"])
                
            except Exception as e:
                print(f"Error extracting relationships: {e}")

            for source, source_type, target, relation, target_type in relationships:
                if source in self.nodes and target in self.nodes:
                    edge = Edge(self.nodes[source], self.nodes[target], relation)
                    if edge not in self.edges:  # Check for duplicate edges
                        self.edges.append(edge)

    def print_knowledge_graph(self):
        print("\nNodes:")
        for node in self.nodes.values():
            print(node.name)

        print("\nEdges:")
        for edge in self.edges:
            print(f"{edge.source_node.name} {edge.relation} {edge.target_node.name}")

    def find_related_companies(self, person_name: str):
        db = self.client["apollo-salesops"]
        collection = db["__kg"]

        pipeline = [
            {
                "$match": {
                    "_id": person_name
                }
            },
            {
                "$graphLookup": {
                    "from": "__kg",
                    "startWith": "$edges.target",
                    "connectFromField": "edges.target",
                    "connectToField": "_id",
                    "as": "related_companies",
                    "depthField": "depth"
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "related_companies._id": 1,
                    "related_companies.type": 1,
                    "related_companies.depth": 1
                }
            }
        ]

        result = collection.aggregate(pipeline)
        return list(result)


knowledge_graph = KnowledgeGraph(documents)
knowledge_graph.create_knowledge_graph()
knowledge_graph.print_knowledge_graph()
knowledge_graph.store_in_mongodb("apollo-salesops", "__kg")

# Test the function
print('print(knowledge_graph.find_related_companies("Steve Jobs"))')
print(knowledge_graph.find_related_companies("Steve Jobs"))
print('print(knowledge_graph.find_related_companies("Elon Musk"))')
print(knowledge_graph.find_related_companies("Elon Musk"))

print("test")
```

## Output

```
Nodes:
Steve Jobs
Apple
Atari
Steve Wozniak
NeXT
Bill Gates
Microsoft
Elon Musk
SpaceX
PayPal
Tesla
Jeff Bezos
Amazon
Blue Origin

Edges:
Steve Jobs founded Apple
Steve Jobs worked at Atari
Steve Wozniak founded Apple
Steve Jobs founded NeXT
Steve Wozniak worked at Apple
Steve Jobs worked at Apple
Bill Gates founded Microsoft
Bill Gates worked at Microsoft
Elon Musk founded SpaceX
Elon Musk founded PayPal
Elon Musk founded Tesla
Jeff Bezos founded Amazon
Jeff Bezos founded Blue Origin
Knowledge graph created and printed.
Storing knowledge graph in MongoDB.
Knowledge graph stored in MongoDB.
Lets begin.
User Prompt: Write a rap about Elon Musk
QUERY UNDERSTANDING: Extract the name of the person in the prompt.
Person: 
Elon Musk
GRAPH TRAVERSAL: Find related companies to the person.
RELATED COMPANIES:
[{'related_companies': [{'_id': 'PayPal', 'type': 'company', 'depth': 0}, {'_id': 'SpaceX', 'type': 'company', 'depth': 0}, {'_id': 'Tesla', 'type': 'company', 'depth': 0}]}]
Contextual Fusion: Combines graph information with textual context.
[
  {
    "role": "system",
    "content": "You are a helpful assistant that uses the provided additional context to generate more relevant responses."
  },
  {
    "role": "user",
    "content": "Given this user prompt: Write a rap about Elon Musk"
  },
  {
    "role": "user",
    "content": "Given this additional context: ```\n[{'related_companies': [{'_id': 'PayPal', 'type': 'company', 'depth': 0}, {'_id': 'SpaceX', 'type': 'company', 'depth': 0}, {'_id': 'Tesla', 'type': 'company', 'depth': 0}]}]\n```"
  },
  {
    "role": "user",
    "content": "\n     Respond to the user prompt in JSON format.\n[response format]\n     - JSON object: { \"response\": \"answer goes here\" }\n"
  }
]
Language Model: Generates human-like text based on provided information.
Yo, let me drop a verse about the man named Elon,
From PayPal streets to space, he's a true phenom.
Started with the bucks, making money on the run,
Now he’s launching rockets, man, ain’t that some fun?

SpaceX in the sky, reppin' the red, white, and blue,
Falcon rockets soaring, making dreams come true.
Starlink in the clouds, internet for the masses,
Connecting all the corners, breaking all the barriers, classes.

Then there's Tesla, take a ride in the future,
Electric dreams rolling, silent like a suturer.
Sustainable vibes, he’s changing up the game,
With each new model, we all look on, stake our claim.

Innovator in the lab, pushing limits to the edge,
With a vision so vivid, he walks that razor's ledge.
To Mars he wants to go, colonize the red, (hey!)
With his mind in the stars, and the world in his thread.

So here's to you, Elon, keep reaching for the sky,
In this rap we celebrate, let your ambitions fly.
From Earth to the stars, with a bright LED glow,
You’re the rocket man, let the whole world know!
```
 
## Conclusion

The knowledge graph stands as a foundational component in the evolution of artificial intelligence, providing a structured framework for representing and connecting information. By serving as a comprehensive repository of entities, attributes, and relationships, the knowledge graph empowers AI systems to reason, learn, and adapt in ways previously unimaginable. 

By integrating graph knowledge we can build AI systems that can dynamically interact with external tools and databases, expanding their problem-solving abilities. Moreover, a shared knowledge graph facilitates collaboration among specialized AI agents, creating intelligent systems capable of addressing complex challenges from multiple perspectives. As knowledge graph technologies continue to mature, we can anticipate the emergence of AI systems that possess an increasingly sophisticated understanding of the world, enabling them to make informed decisions and take actions with human-like nuance. 


