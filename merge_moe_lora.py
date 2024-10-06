from transformers import AutoModelForCausalLM, AutoTokenizer
from openrag.configuration_openrag import OpenRAGConfig
from openrag.modeling_openrag import LlamaForCausalLM
from peft import PeftModel
import torch
import shutil, os, json
import time


def merge_lora_to_base_model(base_model, model_path):
    from openrag.transformers_utils import get_keys_to_not_convert, _load_pretrained_model
    import transformers.utils.bitsandbytes
    import transformers.modeling_utils

    transformers.utils.bitsandbytes.get_keys_to_not_convert = get_keys_to_not_convert
    transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
        _load_pretrained_model
    )
    
    if model_path[-1] != "/": model_path += "/"
    peft_path = model_path + "adapter_model/"
    moe_path = model_path + "moe_model.bin"
    save_path = model_path + "merged/" 
    shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path)
    shutil.copy("./openrag/configuration_openrag.py", save_path+"configuration_openrag.py")
    shutil.copy("./openrag/modeling_openrag.py", save_path+"modeling_openrag.py")
    time.sleep(3)
    
    moe_weights = torch.load(
        moe_path, map_location=torch.device("cuda:0"))
    print("Moe loaded")
    
    print("Loading tokenizer")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True
    )
    print("Loaded tokenizer.")

    model_config = OpenRAGConfig.from_pretrained(base_model)
    model_config.pretraining_tp = 1 

    # Place the corresponding two files in the save_path
    model_config.architectures = ["modeling_openrag.LlamaForCausalLM"]
    model_config.auto_map = {
        "AutoConfig": "configuration_openrag.OpenRAGConfig",
        "AutoModelForCausalLM": "modeling_openrag.LlamaForCausalLM",
        "AutoModel": "modeling_openrag.LlamaModel",
    }

    model_config.moe_dtype = "bfloat16"
    model_config.adapter_dim = 512
    model_config.topk = 2
    model_config.moe_scaling = 0.25
    model_config.num_experts = 8
    model_config.output_router_logits = False
    
    print("Loading model.", model_config)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        output_loading_info=True,
    )
    print("Model loaded")
    model = model[0]
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print(embedding_size)
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    
    weights_dict = {}
    for k, v in moe_weights.items():
        _key = k.replace("base_model.model.", "")
        weights_dict[_key] = v
        
    print("Loading Peft...")
    model = PeftModel.from_pretrained(
        model, 
        peft_path, 
    )
    print("PEFT loaded")
    
    model = model.merge_and_unload()
    print("Merge unload done.")
    
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    print("Saving tokenizer.")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)  
        config["architectures"] = ["modeling_openrag.LlamaForCausalLM"]
        config["auto_map"] = {
            "AutoConfig": "modeling_openrag.OpenRAGConfig",
            "AutoModel": "modeling_openrag.LlamaModel",
            "AutoModelForCausalLM": "modeling_openrag.LlamaForCausalLM"
        }
        config["model_type"] = "llama"
        config.pop("_name_or_path", None)
        json.dump(config, open(config_path, "w"), indent=2)
        time.sleep(3)    

def test_generation(model, tokenizer):
    inputs = "### Instruction:\nHow are you?\n\n### Response:\n"
    inputs = tokenizer(
        inputs, return_tensors="pt"
    )
    
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs, 
                          max_length=512, 
                          do_sample=False, 
                          num_return_sequences=1,)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=False))

def test_loading(model_path):    
    if model_path[-1] != "/": model_path += "/"
    path = model_path + "merged/"

    tokenizer = AutoTokenizer.from_pretrained(path)
    print("Tokenizer loaded.")
    model = AutoModelForCausalLM.from_pretrained(
        path, 
        device_map="cuda:0", 
        trust_remote_code=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params/(1_000_000_000):.2f} B total parameters.")
    # embedding_size = model.get_input_embeddings().weight.shape[0]
    test_generation(model, tokenizer)
    

# pip install 'transformers==4.36.2'
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Script description here")
    parser.add_argument("--base_model", type=str, 
                        default="meta-llama/Llama-2-13b-hf", help="Base model to use")
    parser.add_argument("--model_path", type=str, default="./checkpoints/", 
                        help="Path to model checkpoints")
    args = parser.parse_args()
    merge_lora_to_base_model(args.base_model, args.model_path)
    test_loading(args.model_path)
