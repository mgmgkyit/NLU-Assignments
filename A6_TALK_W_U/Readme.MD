# Let's Talk with Someone : (RAG)

By : Maung Maung Tyi Tha (st125214)

In this task, I focuses on Practical applications of RAG (Retrieval-Augmented Generation) techniques in Langchain framework to augment the chatbot that specializes in answering questions related to a person.

I experimented the text generation with different LLM models, evaluate their performance, resource demand, and task feasibility.

---

## Source Discovery

I decided to use a person, Mr. Bill Gates, who is a funder of Microsoft Corporation.
I used two documents, one is Mr. Bill Gates' Resume, which is curated and created from his public domain information.
The other document is Mr. Bill Gates' information from Wikipaedia, which I printed as a pdf file.

- Bill Gates Resume.pdf
- Bill Gates - Wikipedia.pdf

## Chatbot Prompt
I designed the chatbot prompt as below, with an instruction, Context, Question and Answer.

            """Answer the following question based solely on the provided context. 
            If the answer is not present in the context, say 'I don't know.'

            Context: {context}

            Question: {question}

            Answer:"""

## Exploration and Experimenation of Text Generation Models
The below models were explored in this task.

(a) Successfully deployed
1. TinyLLM
2. Phi-2
3. GPT-2 Medium

(b) Deployment Unsuccessful
1. OpenAI Models
    OpenAI models can be used through OpenAI API Key, but paid subscription on OpenAI services is required for applying OpenAI models.
    My OpenAI account have no subscription, and I am unable to use OpenAI models.
2. Models that based on 7B or more parameters (eg: Zephyr-7B, Mistral-7B )
    Those models were available from HuggingFace, but the demand for GPU and VRAM resource is more than my work environment.

## Analysis and Problem Solving

###List of Retriever and Generator Models

(a) Retriever model - 'hkunlp/instructor-base' 
    It's an instruction-finetuned text embedding model that can generate text embeddings tailored to any task.
    - source : 'https://huggingface.co/hkunlp/instructor-base'

(b) Generator Models :

    1. TinyLLama        : TinyLlama/TinyLlama-1.1B-Chat-v1.0
        Source          : 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        Model Size      : 1.1B params
        Tensor Type     : BF16 (16 bit floating point)

    2. Phi-2            : microsoft/phi-2
        Source          : 'https://huggingface.co/microsoft/phi-2'
        Model Size      : 2.78B params
        Tensor Type     : BF16 (16 bit floating point)

    3. GPT-2 Medium     : openai-community/gpt2-medium
        Source          : 'https://huggingface.co/openai-community/gpt2-medium'
        Model Size      : 380M params
        Tensor Type     : F32 (32 bit floating point)

(c) Codes for experimentation with different Generator Models
    1. TinyLLama    - rag-langchain-st125214-tinyllama.ipynb
    2. Phi-2        - rag-langchain-st125214-phi2.ipynb
    3. GPT-2 Medium - rag-langchain-st125214-gpt2.ipynb


### Issues related to the models providing unrelated information

When unrelated information was provided, the models give the answer as 'undefined'.

Other Issues found were :

- TinyLLM Model has the better overall answering
- Phi-2 Model cannot answer some of the questions
- GPT2-Medium can answer the questions, but not complete

## Generated Question-Answer Pairs

From each experiment code, 11 questions were feed into the generator model, and the question:answer pairs were saved into separate JSON files.

Question-answer pairs will be saved in JSON format, under the folder Answers JSON. The JSON format is as follows:

[
    {
        "question": "What is your name?",
        "answer": "Bill Gates",
    },
...
]

## Chatbot App Development

Among the explored models, the results of TinyLLM seems to have the best output.
Therefore, the app is coded to use TinyLLM as Generator model.

1. Run the **app.py** file.
2. Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
3. Enter a prompt and generate a response.