# A5: Optimization Human Preference - using Direct Preference Optimization (DPO) trainer

By : Maung Maung Tyi Tha (st125214)

In this task, I focuses on using Hugging Face models to optimize human preference, specifically leveraging the Direct Preference Optimization (DPO) trainer. 
I've  to work with preference dataset, train a model, and push it to the Hugging Face model hub. Additionally, Later build a simple web application to 
demonstrate the trained model.

---

## Task 1 : Data Preparation

For the preference dataset, I applied jondurbin/truthy-dpo-v0.1 dataset. 
The dataset jondurbin/truthy-dpo-v0.1 is designed to enhance the truthfulness of large language models (LLMs) without compromising their ability to role-play as humans. It primarily targets areas such as corporeal, spatial, temporal awareness, and common misconceptions.​

Key Features:
    Format: Parquet​
    Size: Between 1,000 and 10,000 samples​
    License: CC BY 4.0​

This dataset has been employed in fine-tuning models to improve their truthfulness. For instance, a user reported positive results after training a model using only the first 200 cases due to hardware constraints.

For my case, I choose sentences from dataset which have character counts only between (30 ~ 300) to overcome resource limitation. I also used first 7 from the dataset to scale the process workable on my environment.

https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1

---


## Task 2 – Model Fine-Tuning with DPOTrainer

### Model & Dataset

The Qwen2-0.5B-Instruct model is an instruction-tuned language model developed by Alibaba Cloud. It is part of the Qwen2 series, which includes models of various sizes designed for tasks such as language understanding, generation, and multilingual applications. The Qwen2-0.5B-Instruct model specifically contains approximately 0.5 billion parameters and is fine-tuned to follow instructions effectively. It incorporates architectural features like the Transformer structure with SwiGLU activation, attention QKV bias, and group query attention. Additionally, it utilizes an improved tokenizer that adapts to multiple natural languages and code. The model has been pretrained on a large dataset and further refined through supervised fine-tuning and direct preference optimization. For optimal performance, it is recommended to use this model with Hugging Face's transformers library version 4.37.0 or later.

https://huggingface.co/Qwen/Qwen2-0.5B-Instruct

- **Model:** Qwen/Qwen2-0.5B-Instruct  
- **Dataset:** Preprocessed version of jondurbin/truthy-dpo-v0.1 with `prompt`, `chosen`, and `rejected` fields.

### Training Parameters & Stages

Main Training Parameters : 
   learning_rate  = [1e-5]     
   batch_sizes    = [5]           
   num_epochs     = [5]           
   betas          = [0.1] 

beta is the temperature parameter that controls how strongly the preference signal is weighted in DPO training
   beta = 0.0 corresponds to maximum likelihood training
   beta = 1.0 corresponds to maximum preference training

- **Training Stages:**
  1. **Model Loading:**  
     The pre-trained model is loaded along with a reference model (a copy of the same model). Both are moved to GPU.
  
  2. **Configuration:**  
     A `DPOConfig` object is used to specify training parameters (including the beta parameter) and to manage aspects like gradient accumulation and fp16 training.
  
  3. **Fine-Tuning:**  
     The `DPOTrainer` computes the DPO loss on each batch, updates the model weights accordingly, and logs training progress.
  
  4. **Hyperparameter Search:**  
     A loop over various hyperparameters (learning rate, batch size, number of epochs, and beta) was implemented to find the best training configuration.

- **Model Training Results**
![Training](screenshots/Model_training.png)
---

## Task 3 – Saving and Pushing the Model

### Process
After training, both the model and the tokenizer are saved locally. The next steps involve pushing these artifacts to the Hugging Face Hub.

- **Steps:**
  1. **Local Saving:**  
     The fine-tuned model and tokenizer are saved to the directory `./dpo_finetuned_model`.
  
  2. **Hub Upload:**  
     Using the `push_to_hub()` method, both the model and tokenizer are uploaded to the repository:
     
     **My HuggingFace Repository ID:** `mgmgkyit/dpo_finetuned_model`
     
     This makes the model publicly available for inference and further use.

---

## Task 4 – Web Application for Testing and Inference

### Web App Overview
A simple web application was developed using Flask. The application provides an interactive interface where users can input a prompt and receive a generated response from the fine-tuned model.

### Webapp Home page and Result
- **Home Page**
![Initial](./screenshots/Initial_Screen.png)  

- **Results**
![Result](screenshots/Answer_1.png)
![Result](screenshots/Answer_2.png)
![Result](screenshots/Answer_3.png)

- **Key Features:**
  - **Input:** A text box for entering a prompt.
  - **Generate Button:** A button to trigger the model’s inference.
  - **Output:** A text box that displays the model-generated response.

### How It Works
1. **Model Loading:**  
   The application loads the model and tokenizer from the Hugging Face Hub using the repository ID.
  
2. **Response Generation:**  
   The function `generate_response()` tokenizes the user input, generates a response using the model’s `generate()` function, and then decodes and displays the output.
  
3. **Interface:**  
   Gradio is used to build a simple, intuitive web interface with an input box, generate button, and an output display. A footer is included to show the developer’s name.

---

## Running the Project

### Training and Evaluation
1. Open the provided Jupyter Notebook and run the cells corresponding to Tasks 1–3 to prepare data, fine-tune the model, and push the model to the Hugging Face Hub.
   ( you will need a huggingface account and API key login to be able to push the model )
2. Verify that the model is successfully uploaded by checking the repository: `mgmgkyit/dpo_finetuned_model`.

### Web Application
1. Run the **app.py** file.
2. Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
3. Enter a prompt and generate a response.

---
