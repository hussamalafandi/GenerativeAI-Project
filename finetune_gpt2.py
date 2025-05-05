from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
import os

def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def main():
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Wichtig für Trainer

    model = GPT2LMHeadModel.from_pretrained(model_name)

    train_dataset = load_dataset("input.txt", tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=1,
        logging_steps=100,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./gpt2-finetuned")
    tokenizer.save_pretrained("./gpt2-finetuned")

if __name__ == "__main__":
    main()
