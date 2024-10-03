from datasets import load_dataset

def preprocess_flipped(cfg, tokenizer):
    # Load the SQuAD dataset based on the cfg
    if cfg.dataset.name == 'squad_v2':
        dataset = load_dataset("squad_v2")
    else:
        dataset = load_dataset("squad")
    
    # Split the original training set into a new training and validation set
    train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    validation_dataset = train_test_split['test']

    # Use the original validation set as the test set
    test_dataset = dataset['validation']

    # Tokenize the datasets
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        answers = [a['text'][0].strip() if len(a['text']) > 0 else "" for a in examples["answers"]]

        inputs = [f"question: {q} context: {c} answer:" for q, c in zip(questions, contexts)]
        labels = [f"question: {q} context: {c} answer: {a}" for q, c, a in zip(questions, contexts, answers)]
        
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        tokenized_labels = tokenizer(labels, max_length=512, truncation=True, padding="max_length")

        model_inputs["labels"] = tokenized_labels["input_ids"]

        return model_inputs

    # Process and save the tokenized datasets
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_validation = validation_dataset.map(preprocess_function, batched=True, remove_columns=validation_dataset.column_names)

    def preprocess_test_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        prompts = [f"question: {q} context: {c} answer:" for q, c in zip(questions, contexts)]
        inputs = tokenizer(prompts, max_length=512, truncation=True, padding=True)
        inputs["answers"] = examples["answers"]
        inputs["id"] = examples["id"]
        return inputs

    tokenized_test = test_dataset.map(preprocess_test_function, batched=True, remove_columns=['title', 'context', 'question'])
    tokenized_validation_eval = validation_dataset.map(preprocess_test_function, batched=True, remove_columns=['title', 'context', 'question'])

    print(f"Columns in the training dataset: {tokenized_train.column_names}")
    print(f"Saving tokenized training dataset to {cfg.dataset.data_path}/train")
    tokenized_train.save_to_disk(f'{cfg.dataset.data_path}/train')
    print(f"Number of rows in the training dataset: {len(tokenized_train)}")
    
    print(f"Columns in the validation dataset: {tokenized_validation.column_names}")
    print(f"Saving validation dataset to {cfg.dataset.data_path}/validation")
    tokenized_validation.save_to_disk(f'{cfg.dataset.data_path}/validation')
    print(f"Number of rows in the validation dataset: {len(tokenized_validation)}")
    
    print(f"Columns in the validation eval dataset: {tokenized_validation_eval.column_names}")
    print(f"Saving validation eval dataset to {cfg.dataset.data_path}/val_eval")
    tokenized_validation_eval.save_to_disk(f'{cfg.dataset.data_path}/val_eval')
    print(f"Number of rows in the validation eval dataset: {len(tokenized_validation_eval)}")

    print(f"Columns in the test dataset: {tokenized_test.column_names}")
    print(f"Saving test dataset to {cfg.dataset.data_path}/test")
    tokenized_test.save_to_disk(f'{cfg.dataset.data_path}/test')
    print(f"Number of rows in the test dataset: {len(tokenized_test)}")

    print("Datasets saved!")