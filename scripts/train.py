from torch.nn import CrossEntropyLoss
from transformers import get_scheduler, DataCollatorWithPadding
from tqdm import tqdm
from torch.optim import AdamW
import os, torch, random
from dotenv import load_dotenv
from datasets import load_from_disk
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf
import time
from models.model_utils import get_model, get_tokenizer
from scripts.evaluate import evaluate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_answer_mask(input_ids, tokenizer):
    batch_size, seq_length = input_ids.shape
    input_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
    answer_start_positions = input_lengths - 1 
    answer_mask = torch.zeros((batch_size, seq_length - 1), device=input_ids.device, dtype=torch.bool)

    for i in range(batch_size):
        answer_mask[i, answer_start_positions[i] - 1:] = 1

    return answer_mask

def create_random_mask(input_ids, tokenizer, mask_probability=0.15):
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in input_ids.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    
    mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        candidate_indices = [j for j in range(seq_length) if not special_tokens_mask[i][j]]
        num_to_mask = int(len(candidate_indices) * mask_probability)
        masked_indices = random.sample(candidate_indices, num_to_mask)
        mask[i, masked_indices] = False
    
    return mask



def decode_and_color_mask(input_ids, mask, tokenizer):
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    colored_text = ""
    toggle = mask[0].item()

    for index in range(len(decoded_tokens)):
        token = tokenizer.convert_tokens_to_string([decoded_tokens[index]])
        color = GREEN if toggle else YELLOW
        colored_text += f"{color}{token}{RESET}"
        if index < len(decoded_tokens) - 1 and index < len(mask) - 1 and mask[index] != mask[index + 1]:
            toggle = not toggle

    return colored_text.strip()

load_dotenv()

def train(cfg):
    tokenizer = get_tokenizer(cfg)
    model = get_model(cfg, tokenizer)
    param_count = count_parameters(model)
    cfg.param_count = param_count

    # Initialize wandb
    wandb_username = os.getenv("WANDB_USERNAME")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run_name = f"{cfg.model.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    tags = wandb_config.pop('tags', None)
    if tags is not None and isinstance(tags, str):
        tags = tags.split(',')
    tags.append(run_name)

    # Initialize Accelerator with wandb logging and gradient accumulation
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=cfg.run.gradient_accumulation_steps,
        )
    accelerator.init_trackers(
        project_name=wandb_project, 
        config=wandb_config,
        init_kwargs={
            "wandb": {
                "entity": wandb_username,
                "name": run_name + "_train",
                "group": cfg.group,
                "tags": tags
            }
        }
    )
    
    # Only print on the main process
    if accelerator.is_main_process:
        print("^"*100)
        print(f"Number of parameters of the model: {param_count}")
        print("^"*100)
        print(OmegaConf.to_yaml(cfg))
    
    # Load preprocessed dataset
    train_dataset = load_from_disk(os.path.join(cfg.dataset.data_path, 'train'))
    train_dataset.set_format(type='torch')
    validation_dataset = load_from_disk(os.path.join(cfg.dataset.data_path, 'validation'))
    validation_dataset.set_format(type='torch')

    # Limit the number of training samples if specified
    if cfg.run.samples is not None:
        print(f"Limiting the number of training samples to {cfg.run.samples}")
        train_dataset = train_dataset.select(range(cfg.run.samples))
        validation_dataset = validation_dataset.select(range(cfg.run.samples))

    # Create data loaders
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=cfg.run.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.run.batch_size, shuffle=False, collate_fn=collate_fn)

    # Prepare optimizer and lr_scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.run.lr, weight_decay=0.1)

    num_train_epochs = cfg.run.epochs
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * num_train_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )

    # Prepare all components with Accelerator
    model, optimizer, train_loader, validation_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, validation_loader, lr_scheduler
    )
    
    accelerator.print(f"Number of update steps: {num_training_steps}")
    accelerator.print(f"Number of batches: {steps_per_epoch}")
    accelerator.print(f"Number of rows in the training dataset: {len(train_dataset)}")
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    patience = cfg.patience
    accelerator.print(f"Training on {accelerator.device}")
    
    for epoch in range(num_train_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(
            enumerate(train_loader, start=1),
            total=steps_per_epoch,
            desc=f'Epoch {epoch + 1}/{num_train_epochs}',
            disable=not accelerator.is_local_main_process
        )
        model.train()
        for step, batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                logits = outputs.logits
                
                if cfg.loss_strategy == "masking":
                    # Shift the logits and labels for autoregressive prediction
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = batch['labels'][:, 1:].contiguous()

                    # Create the answer mask from the shifted labels
                    answer_mask = create_answer_mask(batch['input_ids'], tokenizer)
                    # Apply the answer mask 
                    masked_labels = shift_labels.masked_fill(~answer_mask, tokenizer.pad_token_id)
                    shift_logits = shift_logits.masked_fill(~answer_mask[:, :, None], tokenizer.pad_token_id)

                    if step == 1:
                        print("\nAfter masking:")
                        print("\nFirst sequence (after masking, yellow text is masked):\n")
                        colored_text = decode_and_color_mask(batch['labels'][0], answer_mask[0], tokenizer)
                        print(colored_text)
                        decoded_masked_labels = tokenizer.decode(masked_labels[0], skip_special_tokens=False)
                        print(f"Decoded Masked Labels: \n{decoded_masked_labels}\n")
                        decoded_labels = tokenizer.decode(batch['labels'][0], skip_special_tokens=False)
                        print(f"Decoded Labels: \n{decoded_labels}\n")
                        predicted_token_ids = torch.argmax(shift_logits[0], dim=-1)
                        decoded_predictions = tokenizer.decode(predicted_token_ids, skip_special_tokens=False)
                        print(f"Model Output (Predictions): \n{decoded_predictions}\n")
                        # print the input sequence
                        input_sequence = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
                        print(f"Input Sequence: \n{input_sequence}\n")
                        
                        print("\nAfter masking:")
                        print("\nFirst sequence (after masking, yellow text is masked):\n")
                        colored_text = decode_and_color_mask(batch['labels'][1], answer_mask[1], tokenizer)
                        print(colored_text)
                        decoded_masked_labels = tokenizer.decode(masked_labels[1], skip_special_tokens=False)
                        print(f"Decoded Masked Labels: \n{decoded_masked_labels}\n")
                        decoded_labels = tokenizer.decode(batch['labels'][1], skip_special_tokens=False)
                        print(f"Decoded Labels: \n{decoded_labels}\n")
                        predicted_token_ids = torch.argmax(shift_logits[1], dim=-1)
                        decoded_predictions = tokenizer.decode(predicted_token_ids, skip_special_tokens=False)
                        print(f"Model Output (Predictions): \n{decoded_predictions}\n")
                elif cfg.loss_strategy == "random_masking":
                    # New random masking loss calculation
                    random_mask = create_random_mask(batch['input_ids'], tokenizer)
                    masked_input_ids = batch['input_ids'].clone()
                    masked_input_ids[~random_mask] = tokenizer.mask_token_id

                    masked_outputs = model(input_ids=masked_input_ids, attention_mask=batch['attention_mask'])
                    masked_logits = masked_outputs.logits

                    random_mask_labels = batch['input_ids'].clone()
                    random_mask_labels[random_mask] = -100  # Ignore loss for non-masked tokens

                    original_loss = 0
                    random_mask_loss = loss_fn(masked_logits.view(-1, masked_logits.size(-1)), random_mask_labels.view(-1))

                    loss = original_loss + random_mask_loss
                else:
                    # Standard loss calculation (no masking)
                    shift_logits = logits[:, :-1, :].contiguous()
                    masked_labels = batch['labels'][:, 1:].contiguous()
                
                loss = loss_fn(shift_logits.view(-1, logits.size(-1)), masked_labels.view(-1))
                total_loss += loss.item()
                num_batches += 1
                
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'Loss': loss.item()})

        if epoch % cfg.run.log_interval == 0:
            avg_epoch_loss = total_loss / num_batches
            val_loss = evaluate_validation_loss(cfg, model, validation_loader, loss_fn, accelerator, tokenizer)

            # Early stopping and model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save the model snapshot with the best validation loss
                accelerator.print(f"New best validation loss: {best_val_loss} at epoch {best_epoch}. Saving model...")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                path = f"./models/checkpoints/model_{cfg.model.model_type}_dataset_{cfg.dataset.name}_name_{run_name}/"
                unwrapped_model.save_pretrained(
                    path,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
            else:
                epochs_without_improvement += 1
            
            # Log the relevant metrics together
            metrics_to_log = {
                "avg_train_loss": avg_epoch_loss,
                "avg_val_loss": val_loss,
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }

            accelerator.log(metrics_to_log)
            metrics_str = "\n".join([f"{key}: {value:.4f}" for key, value in metrics_to_log.items()])
            accelerator.print(f"Epoch {epoch + 1} Metrics:\n{metrics_str}\n{'='*50}")

            # Check if early stopping criteria is met
            if epochs_without_improvement >= patience:
                accelerator.print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
                break

    accelerator.log({"final_best_val_loss": best_val_loss, "final_best_epoch": best_epoch})
    accelerator.print(f"Model saved to {path}")

    accelerator.print("Training complete!")
    accelerator.end_training()
    return path

def evaluate_validation_loss(cfg, model, validation_loader, loss_fn, accelerator, tokenizer):
    model.eval()
    total_loss = 0
    num_batches = 0
    for batch in tqdm(validation_loader, desc="Evaluating Validation Loss", disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            if cfg.loss_strategy == "masking":
                # Shift the logits and labels for autoregressive prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = batch['labels'][:, 1:].contiguous()
                # Create the answer mask from the shifted labels
                answer_mask = create_answer_mask(batch['input_ids'], tokenizer)
                # Apply the answer mask 
                masked_labels = shift_labels.masked_fill(~answer_mask, tokenizer.pad_token_id)
                shift_logits = shift_logits.masked_fill(~answer_mask[:, :, None], tokenizer.pad_token_id)
            elif cfg.loss_strategy == "random_masking":
                # New random masking loss calculation
                random_mask = create_random_mask(batch['input_ids'], tokenizer)
                masked_input_ids = batch['input_ids'].clone()
                masked_input_ids[~random_mask] = tokenizer.mask_token_id
                
                masked_outputs = model(input_ids=masked_input_ids, attention_mask=batch['attention_mask'])
                masked_logits = masked_outputs.logits
                
                random_mask_labels = batch['input_ids'].clone()
                random_mask_labels[random_mask] = -100  # Ignore loss for non-masked tokens
                
                original_loss = 0
                random_mask_loss = loss_fn(masked_logits.view(-1, masked_logits.size(-1)), random_mask_labels.view(-1))
                
                loss = original_loss + random_mask_loss
            else:
                # Standard loss calculation (no masking)
                shift_logits = logits[:, :-1, :].contiguous()
                masked_labels = batch['labels'][:, 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, logits.size(-1)), masked_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    # Gather the total loss from all processes
    total_loss = accelerator.gather(torch.tensor(total_loss, device=accelerator.device)).sum()
    num_batches = accelerator.gather(torch.tensor(num_batches, device=accelerator.device)).sum()
    
    avg_val_loss = total_loss / num_batches
    
    return avg_val_loss.item()
