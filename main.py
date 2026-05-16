import json
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = input("Model Name: ")

os.makedirs(f"outputs/{model_name}", exist_ok=True)

with open("token.txt", "r") as f:
    token = f.read().strip()
with open("metadata.json", "r") as f:
    metadata = json.load(f)

model_id = f"google/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,  # Changed from torch_dtype
    device_map="auto",
    token=token,
)

DOMAINS = ['LOGIC', 'SENTIMENT', 'POLITIC', 'MORAL']

for domain in DOMAINS:
    for target_layer in metadata[model_name][domain]["layers"]:

        input_file = f"vectors/{model_name}/{domain}/{metadata[model_name][domain][target_layer]}/delta_h_init.npy"
        if not os.path.exists(input_file):
            input_file = f"vectors/{model_name}/{domain}/{metadata[model_name][domain][target_layer]}/delta_h.npy"
        else:
            print("Yee Haw!")

        delta_h_np = np.load(input_file)

        DATA = {
            "baseline": []
        }

        print(f"Model: {model_name}")
        print(f"Domain: {domain}")
        print(f"Target Layer: {target_layer}")

        OUTPUT_FILE = open(
            f"outputs/{model_name}/{domain}_{target_layer}_results.md", "w")

        with open(f"prompts/{domain.lower()}-prompts.txt", "r") as f:
            prompts = [line.strip() for line in f if line.strip()]


        steering_vector = torch.from_numpy(delta_h_np).to(
            model.device, dtype=torch.bfloat16)


        def create_steering_hook(alpha):
            def hook(module, input, output):
                # Apply: hidden_states + (alpha * steering_vector)
                if isinstance(output, tuple):
                    modified_hidden_states = output[0] + (alpha * steering_vector)
                    return (modified_hidden_states,) + output[1:]
                return output + (alpha * steering_vector)
            return hook


        for prompt in prompts: 
            print(f"Processing prompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            alpha_values = range(0, 11)  

            OUTPUT_FILE.write(f"# {prompt}\n")

            for alpha in alpha_values:
                # Register hook with current alpha
                target_alpha = alpha * 2/10  # Scale alpha to the range [0, 2]
                print(f"Running with alpha={target_alpha:.2f}")

                handle = None
                if alpha != 0:
                    handle = model.model.layers[int(target_layer)].register_forward_hook(
                        create_steering_hook(target_alpha))

                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.4,
                            pad_token_id=tokenizer.eos_token_id
                        )

                    generated_text = tokenizer.decode(
                        outputs[0], skip_special_tokens=True)
                    if alpha == 0:
                        DATA["baseline"].append({
                            "original_input": prompt,
                            "generated": generated_text
                        })
                    else:
                        if str(target_alpha) not in DATA:
                            DATA[str(target_alpha)] = []
                        DATA[str(target_alpha)].append({
                            "original_input": prompt,
                            "generated": generated_text
                        })

                    OUTPUT_FILE.write(f"## Alpha {target_alpha}:\n\n")
                    OUTPUT_FILE.write(f"{generated_text.strip()}\n\n")

                finally:
                    # Clean up the hook before the next iteration
                    if handle: handle.remove()

        OUTPUT_FILE.close()
        print("Steering sweep completed. Results saved to file.")

        with open(f"outputs/{model_name}/{domain}_{target_layer}_results.json", "w") as json_file:
            json.dump(DATA, json_file, indent=4)
