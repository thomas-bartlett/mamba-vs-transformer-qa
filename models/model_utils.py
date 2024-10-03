from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, AutoTokenizer
from models.bidirectional_mamba import MambaConfig, MambaLMHeadModel

def get_model(cfg, tokenizer):
    model = None
    
    if cfg.model.model_type == "gpt-neox" or cfg.model.model_type == "pythia":
        if cfg.model.checkpoint:
            print("Loading model from checkpoint: ", cfg.model.checkpoint)
            model = GPTNeoXForCausalLM.from_pretrained(cfg.model.checkpoint)
        elif cfg.task == "train" or cfg.task == "size":
            config = GPTNeoXConfig(
                bos_token_id=0,
                eos_token_id=0,
                hidden_size=cfg.model.hidden_size,
                intermediate_size=cfg.model.hidden_size * 4,
                num_attention_heads=cfg.model.heads,
                num_hidden_layers=cfg.model.layers,
                vocab_size=len(tokenizer),
            )
            model = GPTNeoXForCausalLM(config)
        else:
            raise ValueError("Invalid model configuration")

    elif cfg.model.model_type == "mamba":
        if cfg.model.checkpoint:
            print("Loading model from checkpoint: ", cfg.model.checkpoint)
            model = MambaLMHeadModel.from_pretrained(cfg.model.checkpoint)
        elif cfg.task == "train" or cfg.task == "size":
            config = MambaConfig(
                model_type=cfg.model.model_type,
                d_model=cfg.model.hidden_size,
                n_layer=cfg.model.layers,
                ssm_cfg={"d_state": cfg.model.state_dim},
                vocab_size=len(tokenizer),
            )
            model = MambaLMHeadModel(config)
        else:
            raise ValueError("Invalid model configuration")

    elif cfg.model.model_type == "bidirectional_mamba":
        if cfg.model.checkpoint and ("/models/checkpoints/" in cfg.model.checkpoint):
            print("Loading model from checkpoint: ", cfg.model.checkpoint)
            model = MambaLMHeadModel.from_pretrained(cfg.model.checkpoint)
        elif cfg.model.checkpoint:
            print("Loading model from checkpoint: ", cfg.model.checkpoint)
            model = MambaLMHeadModel.from_pretrained_bidirectional(cfg.model.checkpoint, load_into=cfg.load_into)
        elif cfg.task == "train" or cfg.task == "size":
            config = MambaConfig(
                model_type=cfg.model.model_type,
                d_model=cfg.model.hidden_size,
                n_layer=cfg.model.layers,
                ssm_cfg={"d_state": cfg.model.state_dim},
                vocab_size=len(tokenizer),
            )
            model = MambaLMHeadModel(config)
        else:
            raise ValueError("Invalid model configuration")
    else:
        raise ValueError("Invalid model name")
    
    # Resize token embeddings if new tokens have been added
    if model is not None and hasattr(model, 'resize_token_embeddings'):
        model.resize_token_embeddings(len(tokenizer))
    
    print(model)
    
    return model

def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if cfg.loss_strategy == "random_masking":
        tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    return tokenizer
