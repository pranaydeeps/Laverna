from datasets import load_dataset
from transformers import set_seed
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch

SEED = 42
FILE_NAME = "example.txt"
MODEL = "pranaydeeps/swahbert-base-cased"
INFERENCE_BATCH_SIZE = 8

set_seed(SEED)
data = load_dataset("text", data_files={"train":FILE_NAME})
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_data = data.map(
    preprocess_function,
    batched=True,
    num_proc=8,
    remove_columns=data["train"].column_names,
)
tokenized_data.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15,return_tensors='pt')

dataloader = DataLoader(tokenized_data['train'], shuffle=True, batch_size=INFERENCE_BATCH_SIZE, collate_fn=data_collator)


model = AutoModelForMaskedLM.from_pretrained(MODEL)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
to_save = []
for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch).logits
    for index, item in enumerate(batch['input_ids']):
        mask_token_index = torch.where(item==tokenizer.mask_token_id)[0]
        if mask_token_index.nelement() == 0:
            print("SKIPPING NO MASK")
            continue
        for mask in mask_token_index:
            mask_token_logits = outputs[index, int(mask), :]
            to_save.append([item, torch.tensor([int(mask)]), mask_token_logits])

torch.save(to_save, "outputs_{}.pt".format(MODEL.split('/')[-1]))
            

