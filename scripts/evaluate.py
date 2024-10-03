import torch
import json
from datasets import load_from_disk
from transformers import DataCollator
import os, time, collections, re, string
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
from evaluate import load as load_metric
import numpy as np
from accelerate import Accelerator

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

load_dotenv()

def evaluate(cfg, model, tokenizer, from_main=True, accelerator=None):
    if accelerator is None:
        accelerator = Accelerator()

    if from_main:
        wandb_username = os.getenv("WANDB_USERNAME")
        wandb_project = os.getenv("WANDB_PROJECT")
        wandb_config = OmegaConf.to_container(cfg, resolve=True)

        # Extract the run_name from the model checkpoint path
        checkpoint_path = cfg.model.checkpoint
        if "_name_" in checkpoint_path:
            run_name = checkpoint_path.split("_name_")[-1].strip("/")
        else:
            run_name = checkpoint_path.replace("/", "_")

        tags = wandb_config.pop('tags', None)
        if tags is not None and isinstance(tags, str):
            tags = tags.split(',')
        tags.append(run_name)
        wandb.init(
            project=wandb_project, 
            entity=wandb_username, 
            config=wandb_config,
            name=run_name + "_evaluate",
            group=cfg.group,
            tags=tags
        )
        max_test_samples = cfg.run.max_test_samples
        eval_dataset = cfg.run.eval_dataset
    else:
        max_test_samples = 1000
        eval_dataset = 'val_eval'

    # if eval_dataset == 'val_eval':
    #     print("Loading the validation evaluation dataset.")
    #     dataset = load_from_disk(os.path.join(cfg.dataset.data_path, 'val_eval'))
    # else:
    #     print("Loading test dataset.")
    #     dataset = load_from_disk(os.path.join(cfg.dataset.data_path, 'test'))
    dataset = load_from_disk(os.path.join(cfg.dataset.data_path, 'test'))
    if max_test_samples is not None:
        print(f"Limiting the number of test samples to {max_test_samples}")
        dataset = dataset.select(range(max_test_samples))

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model, data_loader = accelerator.prepare(model, data_loader)

    print("Number of rows in test set:", len(dataset))
    preds = []
    refs = []
    debug_info = []
    f1_scores_list = []

    model.eval()
    print("Generating predictions...")
    total_tokens = 0
    printed_examples = 0
    start_time = time.time()
    for batch in tqdm(data_loader, desc="Generating"):
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(accelerator.device)
        attn_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0).to(accelerator.device)
        # print(f"Input IDs shape: {input_ids.shape}")  # This will now print [1, sequence_length]
        answers = batch["answers"]
        example_id = batch["id"][0]

        max_length = input_ids.shape[1] + 100

        with torch.no_grad():
            with accelerator.autocast():
                if cfg.model.model_type == "mamba" or cfg.model.model_type == "bidirectional_mamba":
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=max_length,
                        cg=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        enable_timing=False,
                        top_k=cfg.run.top_k,
                        temperature=cfg.run.temperature,
                        top_p=cfg.run.top_p,
                    )
                else:
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_length=max_length,
                        return_dict_in_generate=True,
                        pad_token_id=tokenizer.pad_token_id,
                        temperature=cfg.run.temperature,
                        top_k=cfg.run.top_k,
                        top_p=cfg.run.top_p,
                        do_sample=True,
                    )

        input_length = input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        total_tokens += generated_tokens.numel()
        # Print the input and output only for the first 5 examples
        if printed_examples < 5:
            decoded_inputs = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            decoded_outputs = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
            print(f"Input {printed_examples + 1}: {decoded_inputs}")
            print(f"Generated Output {printed_examples + 1}: {decoded_outputs}")
            print("-" * 50)  # Separator for readability
            printed_examples += 1

        
        # Process the single generated output
        pred_model = tokenizer.decode(generated_tokens[0].tolist(), skip_special_tokens=True)
        pred_model = str(pred_model.split("\n\n")[0].strip())
        if "<|endoftext|>" in pred_model:
            pred_model = pred_model.split("<|endoftext|>")[0]

        pred = {"id": example_id, "prediction_text": pred_model}
        if cfg.dataset.name == 'squad_v2':
            pred["no_answer_probability"] = 0.0
        preds.append(pred)
        # print(f"\n\n\nanswers['text']: {answers['text']}")

        refs.append({"id": example_id, "answers": {"text": [ans[0] for ans in answers['text']], "answer_start": [0]}})

        # Determine if the question is answerable (i.e., if there are non-empty gold answers)
        is_answerable = any(ans[0].strip() for ans in answers['text'] if isinstance(ans[0], str))

        # Calculate F1 score for this prediction
        f1_scores = [compute_f1(pred_model, answer[0]) for answer in answers['text']]
        max_f1 = max(f1_scores) if f1_scores else 0
        f1_scores_list.append(max_f1)

        # Get the context tokens (excluding any special tokens)
        context_tokens = input_ids[0][input_ids[0] != tokenizer.pad_token_id]
        context_token_length = len(context_tokens)

        # Decode the context for readability in the debug info
        context = tokenizer.decode(context_tokens, skip_special_tokens=True)

        debug_entry = {
            "id": example_id,
            "context": context,
            "context_token_length": context_token_length,  # Add this line
            "gold_answers": answers['text'],
            "predicted_answer": pred_model,
            "f1_score": max_f1,
            "is_answerable": is_answerable,
        }

        debug_info.append(debug_entry)

    end_time = time.time()
    total_time = end_time - start_time
    throughput = total_tokens / total_time

    print("Predictions generated!")
    # Calculate average and std of F1 scores
    avg_f1 = np.mean(f1_scores_list)
    std_f1 = np.std(f1_scores_list)

    if from_main:
        print("Evaluating predictions...")

        # Load the SQuAD metric
        if cfg.dataset.name == 'squad_v2':
            squad_metric = load_metric("squad_v2")
        else:
            squad_metric = load_metric("squad")

        # Compute the metric
        results = squad_metric.compute(predictions=preds, references=refs)
        print(json.dumps(results, indent=2))

        print(f"Throughput: {throughput:.2f} tokens/sec")
        wandb.log({"throughput": throughput, "avg_f1": avg_f1, "std_f1": std_f1})
        # Save the predictions and debug info
        results_dir = f'results/{run_name}'
        print(f"Saving results to {results_dir}")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'predictions.json'), 'w') as f:
            json.dump(preds, f)

        with open(os.path.join(results_dir, 'debug_info.json'), 'w') as f:
            json.dump(debug_info, f)

        with open(os.path.join(results_dir, 'eval_metrics.json'), 'w') as f:
            json.dump(results, f)

        wandb.log(results)
        wandb.finish()

    return avg_f1, std_f1