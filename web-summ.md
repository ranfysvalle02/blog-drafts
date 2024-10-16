## **Intro**

**Tired of your language model struggling to grasp the big picture?** Traditional context windows, like a tiny viewing port, limit its ability to see the entire forest. This is especially frustrating when trying to summarize long, complex texts.

Text summarization is a critical task that involves condensing a large volume of text into a concise summary. This blog post will delve into a Python code snippet that employs a map-reduce strategy to summarize web content using a large language model.

Whether you're a researcher, a student, or a professional, the ability to quickly extract and summarize relevant information is a valuable skill. With the help of Python and Azure OpenAI, we can automate this process and save time.

## **Map-Reduce for Text Summarization**

**Understanding the Strategy**

Map-Reduce is a programming model that processes large data sets in parallel. In the context of text summarization, we can break down the task into two steps:

1. **Map:** Divide the input text into smaller chunks, and summarize each chunk individually.
2. **Reduce:** Combine the invidual summaries of these chunks into a final summary.

**Gearing Up for the Adventure**

Before we set off, let's gather our supplies:

1. **Python:** Make sure you have Python installed on your system.
2. **Python Libraries:** We'll need some helpful libraries:
   - `json` and `requests` (standard Python libraries)
   - `concurrent.futures` (for parallel processing)
   - `BeautifulSoup` (for parsing HTML)
   - `openai` (for interacting with Azure OpenAI)
   - `nltk` (for text processing)
3. **Azure OpenAI API Key:** Sign up for an Azure OpenAI account to obtain your own key.

**Unraveling the Code**

Let's dissect the code step-by-step:

**Import Libraries:**

```python
import json
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import tiktoken
```

We begin by importing the necessary libraries. Familiar faces like `json` and `requests` join the crew, along with `concurrent.futures` for multitasking, `BeautifulSoup` for navigating the HTML landscape, `openai` for communication with Azure OpenAI, and `nltk` for text processing.

**Laying the Groundwork:**

```python
# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define constants
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = "" 
az_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY)

```

**The Extraction Expedition:**

```python
r = requests.get('https://thenewstack.io/devs-slash-years-of-work-to-days-with-genai-magic/')
# Parsing the HTML
soup = BeautifulSoup(r.content, 'html.parser')
RAW_HTML = soup.get_text()
# Break the text into chunks (adjust chunk size as needed)
chunk_size = 10000
# check status code for response received
# success code - 200
print(r)
print("total tokens: " + str(len(tokenizer.encode(RAW_HTML))))
chunks = [RAW_HTML[i:i+chunk_size] for i in range(0, len(RAW_HTML), chunk_size)]
print(len(chunks))
```

We use `requests.get()` to fetch the content of the target webpage and then leverage BeautifulSoup's expertise to parse it. The extracted text is then divided into manageable chunks for efficient processing.

**The Art of Summarization:**

```python
# Define function to summarize each chunk
def summarize_chunk(chunk):
    print(f"Starting to process chunk at index {chunks.index(chunk)}; chunk_size=" + str(len(tokenizer.encode(chunk))))
    msgs2send = [
        {"role": "system", "content": "You are a helpful assistant that summarizes the CONTENT of SCRAPED HTML."},
        {"role": "user", "content": """=
Your main objective is to summarize the content in such a way that the user does not have to visit the website.
Please provide a bullet list (-) summary to the best of your ability of the CONTENT in this WEBSITE:

IMPORTANT: PAY CLOSE ATTENTION TO THE CONTENT OF THE WEBSITE. IGNORE THINGS LIKE NAVIGATION, ADS, ETC.
REMEMBER! YOUR GOAL IS TO SUMMARIZE THE CONTENT OF THE WEBSITE IN A WAY THAT THE USER DOES NOT HAVE TO VISIT THE WEBSITE.
[web_content]
         
"""},
        {"role": "user", "content": str(chunk)},
        {"role": "user", "content": """
[response format]
JSON object with `web_summary` key with RAW HTML MARKDOWN STRING.
web_summary should be in bullet list, valid markdown format [RAW HTML MARKDOWN STRING]
Max. 3 sentences per bullet point.
Max. 10 bullet points.
Min. 5 bullet points.
[sample_response]

- bullet_point_1

- bullet_point_2

...
         
- bullet_point_n


"""}
    ]
    ai_response = az_client.chat.completions.create(
        model="gpt-4o",
        messages=msgs2send,
        response_format={"type": "json_object"}
    )
    tmp_sum = json.loads(ai_response.choices[0].message.content.strip())
    print(f"Finished processing chunk at index {chunks.index(chunk)}")
    return tmp_sum['web_summary']
```

This function takes a chunk of text as input, sends it to the LLM for processing, and extracts the returned summary.

**Harnessing the Power of Parallel Processing:**

```python
# Use ThreadPoolExecutor to parallelize the summarization
with concurrent.futures.ThreadPoolExecutor() as executor:
    summaries = list(executor.map(summarize_chunk, chunks))

```

We leverage the `ThreadPoolExecutor` from `concurrent.futures` to parallelize the summarization process across multiple chunks. 

This significantly speeds things up, especially for lengthy webpages.

**Piecing it All Together:**

```python
# Summarize the summaries
summary_of_summaries = az_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role":"system", "content":"You are a helpful assistant who can strategically summarize multiple summaries together into one coherent summary."},
        {"role": "user", "content": "[summaries]"},
        {"role": "user", "content": str(summaries)},
        {"role": "user", "content": """
[response format]
JSON object with `web_summary` key with MARKDOWN STRING.
web_summary should be in bullet list, valid markdown format [web_summary=MARKDOWN STRING]
Max. 3 sentences per bullet point.
Max. 10 bullet points.
Min. 5 bullet points.
[sample_response]

- bullet_point_1

- bullet_point_2

...
         
- bullet_point_n
         
[task]
    - Please provide a complete, comprehensive summary using the individual summaries provided in [summaries]
"""}
    ],
    response_format={"type": "json_object"}
)
print("summary_of_summaries:")
print(
    json.loads(
        summary_of_summaries.choices[0].message.content.strip()
    ).get('web_summary')
)
```

Now that we have individual summaries for each chunk, it's time to assemble the bigger picture. We send the list of summaries to the LLM, and it returns a final, comprehensive summary that captures the essence of the entire webpage.

**The Complete Blueprint**

For your reference, here's the complete code:

```python
import json
import requests
import concurrent.futures
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import tiktoken

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Define constants
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = "" 
az_client = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT,api_version="2023-07-01-preview",api_key=AZURE_OPENAI_API_KEY)

r = requests.get('https://thenewstack.io/devs-slash-years-of-work-to-days-with-genai-magic/')
# Parsing the HTML
soup = BeautifulSoup(r.content, 'html.parser')
RAW_HTML = soup.get_text()
# Break the text into chunks (adjust chunk size as needed)
chunk_size = 10000
# check status code for response received
# success code - 200
print(r)
print("total tokens: " + str(len(tokenizer.encode(RAW_HTML))))
chunks = [RAW_HTML[i:i+chunk_size] for i in range(0, len(RAW_HTML), chunk_size)]
print(len(chunks))
# Define function to summarize each chunk
def summarize_chunk(chunk):
    print(f"Starting to process chunk at index {chunks.index(chunk)}; chunk_size=" + str(len(tokenizer.encode(chunk))))
    msgs2send = [
        {"role": "system", "content": "You are a helpful assistant that summarizes the CONTENT of SCRAPED HTML."},
        {"role": "user", "content": """=
Your main objective is to summarize the content in such a way that the user does not have to visit the website.
Please provide a bullet list (-) summary to the best of your ability of the CONTENT in this WEBSITE:

IMPORTANT: PAY CLOSE ATTENTION TO THE CONTENT OF THE WEBSITE. IGNORE THINGS LIKE NAVIGATION, ADS, ETC.
REMEMBER! YOUR GOAL IS TO SUMMARIZE THE CONTENT OF THE WEBSITE IN A WAY THAT THE USER DOES NOT HAVE TO VISIT THE WEBSITE.
[web_content]
         
"""},
        {"role": "user", "content": str(chunk)},
        {"role": "user", "content": """
[response format]
JSON object with `web_summary` key with RAW HTML MARKDOWN STRING.
web_summary should be in bullet list, valid markdown format [RAW HTML MARKDOWN STRING]
Max. 3 sentences per bullet point.
Max. 10 bullet points.
Min. 5 bullet points.
[sample_response]

- bullet_point_1

- bullet_point_2

...
         
- bullet_point_n


"""}
    ]
    ai_response = az_client.chat.completions.create(
        model="gpt-4o",
        messages=msgs2send,
        response_format={"type": "json_object"}
    )
    tmp_sum = json.loads(ai_response.choices[0].message.content.strip())
    print(f"Finished processing chunk at index {chunks.index(chunk)}")
    return tmp_sum['web_summary']

# Use ThreadPoolExecutor to parallelize the summarization
with concurrent.futures.ThreadPoolExecutor() as executor:
    summaries = list(executor.map(summarize_chunk, chunks))

# Summarize the summaries
summary_of_summaries = az_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role":"system", "content":"You are a helpful assistant who can strategically summarize multiple summaries together into one coherent summary."},
        {"role": "user", "content": "[summaries]"},
        {"role": "user", "content": str(summaries)},
        {"role": "user", "content": """
[response format]
JSON object with `web_summary` key with MARKDOWN STRING.
web_summary should be in bullet list, valid markdown format [web_summary=MARKDOWN STRING]
Max. 3 sentences per bullet point.
Max. 10 bullet points.
Min. 5 bullet points.
[sample_response]

- bullet_point_1

- bullet_point_2

...
         
- bullet_point_n
         
[task]
    - Please provide a complete, comprehensive summary using the individual summaries provided in [summaries]
"""}
    ],
    response_format={"type": "json_object"}
)
print("summary_of_summaries:")
print(
    json.loads(
        summary_of_summaries.choices[0].message.content.strip()
    ).get('web_summary')
)
```

**Navigating Potential Hurdles**

If you encounter any roadblocks during your journey, check these points:

1. **Double-check your Azure OpenAI API key.** Ensure it's correct and has the necessary permissions.
2. **Verify library installations.** Make sure all the required Python libraries are installed on your system.
3. **Scrutinize the webpage URL.** Ensure the URL you're trying to scrape is valid and accessible.

**Further Reading**

* [Python Documentation](https://docs.python.org/3/)
* [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [Azure OpenAI Documentation](https://azure.microsoft.com/en-us/services/openai/)
* [concurrent.futures Documentation](https://docs.python.org/3/library/concurrent.futures.html)

