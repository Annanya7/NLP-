# Question Answering in NLP 

# Classification  Approach: 
The provided training log and approach involve fine-tuning the bert-base-uncased model on a multiple-choice question-answering task, specifically tailored to process each answer choice alongside the question stem and a relevant fact. This setup employs a typical classification approach with BERT, leveraging the model's deep understanding of language context. Each input instance is prepared by concatenating the fact, question stem, and one of the answer choices, enclosed between special [CLS] and [END] tokens. This structure allows BERT to contextualize the answer choice within the framework of the given fact and question.The BERT model is complemented by a linear layer that transforms the [CLS] token's embeddings into logits, which are then passed through a softmax layer to obtain predictions. This process is repeated across all choices, with the model being trained to minimize the cross-entropy loss between its predictions and the actual labels. The training procedure spans three epochs, showcasing progressive improvements in training accuracy and reductions in loss, which indicates effective learning.

# Generative Approach:

# Model Selection and Configuration: 
We used the `GPT2LMHeadModel` from the Hugging Face Transformers library. This model is specifically chosen for its capabilities in handling language modeling and text generation tasks, which are essential for generating responses in a multiple-choice question format.
# Tokenizer and Dataset Preparation: 
The `GPT2Tokenizer` is employed to process the input text. The dataset is prepared into instances that include the context (fact and question stem) and multiple-choice options formatted with specific tokens to facilitate proper segmentation by the model. This structured data is then converted into `GPT2GenerationDataset` instances for the training, validation, and testing phases. Each instance is meticulously prepared to ensure that the model focuses on generating the answer part by setting tokens before the `[ANSWER]` token to -100.
# Training Configuration: 
The code uses the `Trainer` class with specific `TrainingArguments` to fine-tune the model. Important hyperparameters include:
 `num_train_epochs=5`: Specifies that the model should be trained over five epochs.
  `per_device_train_batch_size=8`: Sets the batch size to 8, balancing between computational efficiency and memory usage.
 `logging_steps=300`: This ensures that logs are generated every 300 steps, providing frequent updates on the training progress.
Evaluation Methodology: The evaluation is performed by generating answers using the trained model and then assessing these against the correct answers using the ROUGE-L metric, which evaluates the similarity of the generated text to the actual options. The choice with the highest ROUGE-L score is considered the model’s predicted answer, and accuracy is calculated based on these predictions.
