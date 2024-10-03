import hydra
from omegaconf import DictConfig, OmegaConf
from scripts.train import train
from models.model_utils import get_model, get_tokenizer
from scripts.evaluate import evaluate
from scripts.preprocess import preprocess
from scripts.preprocess_flipped import preprocess_flipped
from scripts.preprocess_original import preprocess_original
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.task == "train":
        if os.path.exists(os.path.join(cfg.dataset.data_path, 'train')):
            if os.listdir(os.path.join(cfg.dataset.data_path, 'train')):
                train(cfg)
            else:
                print("The specified training data path is empty. Please run the preprocess task first.")
        else:
            print("The specified training data path does not exist. Please run the preprocess task first.")
    elif cfg.task == "evaluate":
        tokenizer = get_tokenizer(cfg)
        model = get_model(cfg, tokenizer)
        param_count = count_parameters(model)
        cfg.param_count = param_count
        print("^"*100)
        print(f"Number of parameters of the model: {param_count}")
        print("^"*100)
        print(OmegaConf.to_yaml(cfg))
        evaluate(cfg, model, tokenizer)
    elif cfg.task == "preprocess":
        tokenizer = get_tokenizer(cfg)
        if "flipped" in cfg.dataset.name:
            print("Preprocessing flipped dataset")
            preprocess_flipped(cfg, tokenizer)
        if "original" in cfg.dataset.name:
            print("Preprocessing original dataset")
            preprocess_original(cfg, tokenizer)
        else:
            print("Preprocessing dataset")
            preprocess(cfg, tokenizer)

if __name__ == "__main__":
    main()