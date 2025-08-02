import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r"####\s*(\d+)"
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

def load_gsm8k():
    dataset = load_dataset("NLG/data/GSM8K", "main")
    # x = "Q: " + x[0] + "\n" + "A:"
    dataset = dataset.map(
        lambda e: {
            "x": f'Q: {e["question"]}\nA: ',
            "y": e["answer"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["test"]
    return train_set, validation_set, validation_set


def main(model_name, eval_seed=0, temperature=0.8, bsz=4):
    _, _, test_set = load_gsm8k()
    model = LLM(model=model_name,dtype="bfloat16", seed=eval_seed)
    sampling_params = SamplingParams(
        top_p=0.95, temperature=temperature, max_tokens=1024
    )
    all = 0
    correct = 0
    t = tqdm(range(0, len(test_set), bsz))
    for idx in t:
        # pred_text = model_inference(model, tokenizer, example['x'], model_type, max_target_length=512)
        examples = test_set[idx : idx + bsz]
        prompts = examples["x"]
        outputs = model.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
        for y, output in zip(examples["y"], outputs):
            pred_text = output.outputs[0].text
            gt = extract_num(y)
            pred = extract_num(pred_text)
            correct += int(gt == pred)
            all += 1
            t.set_description(f"Accuracy: {correct/all*100:02f}%")

    print("Acc:", correct / all)
    # append to gsm8k_results.txt (create if not exists)
    if not os.path.exists("gsm8k_results.txt"):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model Acc\n")
    with open("gsm8k_results.txt", "a") as f:
        f.write(
            f"{model_name},eval_seed={eval_seed},temperature={temperature}    {correct/all}\n"
        )

if __name__ == "__main__":
    main("NLG/models/fine_tuned_models/LoRA-S_True_rsLoRA=True_DoRA=False_rank=8_alpha=16_seed=0/", eval_seed=0, temperature=0.8, bsz=4)