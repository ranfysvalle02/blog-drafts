**Title:** Building an Advanced AI Agent with OpenAI, MongoDB, and DuckDuckGo

**Introduction:**

In the world of AI, creating an intelligent agent that can interact with users, maintain conversation history, and utilize tools to perform tasks is a fascinating challenge. 
This blog post will guide you through the process of building such an agent using OpenAI, MongoDB, and DuckDuckGo. This assistant can perform tasks, use tools, and maintain a conversation history.

**The Foundation: Setting Up the Environment**

Our first step is to set up our environment. This involves importing necessary libraries and defining some constants. We're using libraries like `openai` for AI model interaction, `pymongo` for database management, `duckduckgo_search` for web search, and `youtube_transcript_api` for fetching YouTube video transcripts.

```python
import json
from openai import AzureOpenAI
import pymongo
from duckduckgo_search import DDGS
import asyncio
from datetime import datetime
import re 
from youtube_transcript_api import YouTubeTranscriptApi
...
```

**The Memory: Conversation History Management**

As our AI assistant interacts with users, it's crucial to remember past conversations. This memory allows the assistant to provide context-aware responses. We create a `ConversationHistory` class to manage this memory, storing it either in a local list or a MongoDB database for more persistent storage.

```python
class ConversationHistory:
    """
    Class to manage conversation history, either in memory or using MongoDB.
    """
    def __init__(self, mongo_uri=None):
        self.history = []  # Default to in-memory list if no Mongo connection

        # If MongoDB URI is provided, connect to MongoDB
        if mongo_uri:
            self.client = pymongo.MongoClient(mongo_uri)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]

    def add_to_history(self, text, is_user=True):
        """
        Add a new entry to the conversation history.
        """
        timestamp = datetime.now().isoformat()
        # If MongoDB client is available, insert the conversation into MongoDB
        if self.client:
            self.collection.insert_one({"text": text, "is_user": is_user, "timestamp": timestamp})
        else:
            self.history.append((text, is_user, timestamp))

    def get_history(self):
        """
        Get the conversation history as a formatted string.
        """
        if self.client:
            history = self.collection.find({}, sort=[("timestamp", pymongo.DESCENDING)]).limit(2)
            return "\n".join([f"{item['timestamp']} - {'User' if item.get('is_user', False) else 'Assistant'}: {item['text']}" for item in history])
        else:
            return "\n".join([f"{timestamp} - {'User' if is_user else 'Assistant'}: {text}" for text, is_user, timestamp in self.history])
```

**The Skills: Tools and Tasks**

Our AI assistant needs skills to perform tasks. We represent these skills as `Tool` objects. For instance, we have a `SearchTool` that uses the DuckDuckGo search engine to retrieve information from the web. 

```python
class Tool:
    """
    Base class for tools that the agent can use.
    """
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.openai = az_client
        self.model = "gpt-4o"

    def run(self, input):
        """
        This method needs to be implemented by specific tool classes.
        """
        raise NotImplementedError("Tool must implement a run method")

class SearchTool(Tool):
    """
    Tool to search the web using DuckDuckGo.
    """
    def run(self, input):
        """
        Runs a DuckDuckGo search and returns the results.
        """
        results = DDGS().text(str(input+" site:youtube.com video"), region="us-en", max_results=5)
        return {"web_search_results": results, "input": input, "tool_id": "<" + self.name + ">"}

```
**The Actions: Tasks**

Tasks are the actions that our AI assistant can perform using its skills. For example, a task could be to search for information on a specific topic and summarize the findings. Each task is represented as an instance of the `Task` class, which includes a description of the task, the agent that will perform the task, the tools that the agent will use, and the input that the task will process.

**The Brain: Advanced Agent and Custom Process**

The `AdvancedAgent` is the brain of our AI assistant. It orchestrates the use of tools and tasks, generates responses to user prompts, and maintains the conversation history. 

```python
class AdvancedAgent:
    """
    Advanced agent that can use tools and maintain conversation history.
    """
    ...
```
**The Workflow: Custom Process**

Sometimes, our AI assistant needs to perform a series of tasks in a specific order. That's where the `CustomProcess` class comes in. It allows us to chain tasks together to create more complex workflows. For instance, we might have a process that involves searching for information, analyzing the results, and then summarizing the findings.

```python
class CustomProcess:
    """
    Class representing a process that consists of multiple tasks.
    """
    ...
```

**The Adventure: Running the AI Assistant**

With all the components in place, it's time to set our AI assistant in motion. We create tasks, chain them into a process, and set the process running. As the tasks are performed, our AI assistant interacts with the user, remembers the conversation, and uses its tools to generate responses.

```python
async def main():
    # Use a MongoDB connection string for persistent history (optional)
    # Create tasks
    user_input = input("Enter any topic: ")
    task1 = Task(
        description=str("Perform a search for: `"+user_input+"`"),
        agent=AdvancedAgent(
            tools=[SearchTool("search", "Search the web.")],
            history=ConversationHistory(MDB_URI)
        ),
        name="step_1",
        tool_use_required=True
    )
    ...
    # Run process and print the result
    result = await my_process.run()
    print(result[-1].get("answer", ""))

if __name__ == "__main__":
    asyncio.run(main())
```

For instance, if a user inputs the topic "Python programming", the AI assistant will search the web for information on Python programming, summarize the findings, and present a concise report to the user.

Conclusion:

Building an AI assistant with Python is a complex but rewarding task. It involves setting up the environment, crafting memory management systems, developing skills and tasks, and creating an intelligent agent to orchestrate everything. By understanding these components, you can create a flexible and efficient AI assistant that can handle a wide range of tasks. So, roll up your sleeves, and happy coding!
