import os
import yaml
import logging
from ludwig.api import LudwigModel

os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv('HUGGINGFACE_API_KEY')

# Ludwig configuration
config_str = """
model_type: llm
base_model: mistralai/Mistral-7B-v0.1
# base_model: alexsherstinsky/Mistral-7B-v0.1-sharded
# base_model: Siddharthvij10/MistralSharded2
quantization:
  bits: 4
adapter:
  type: lora
prompt:
  template: |
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:
input_features:
  - name: prompt
    type: text
    preprocessing:
      max_sequence_length: 256
output_features:
  - name: output
    type: text
    preprocessing:
      max_sequence_length: 256
trainer:
  type: finetune
  learning_rate: 0.0001
  batch_size: 1
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate_scheduler:
    warmup_fraction: 0.01
preprocessing:
  sample_ratio: 0.1
"""
config = yaml.safe_load(config_str)

# Train model
model = LudwigModel(config=config, logging_level=logging.INFO)
results = model.train(dataset="tatsu-lab/alpaca_farm")

# Save the model
model.save("results")


#Run this command to upload your Trained Model to Hugging Face with name as "goutham/mistralai-7B-v01-fine-tuned-using-ludwig-4bit"
python -m ludwig.upload hf_hub --repo_id "goutham/mistralai-7B-v01-fine-tuned-using-ludwig-4bit" --model_path results/api_experiment_run_2