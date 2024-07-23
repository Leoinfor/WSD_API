import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pytorch_lightning as pl
import transformers
import spacy
import re
nlp = spacy.load("en_core_web_sm")

# class T5FineTuner(pl.LightningModule):
#   def __init__(self, model_name, batch_size=8):
#     super(T5FineTuner, self).__init__()
#     #self.save_hyperparameters()
#     self.model = T5ForConditionalGeneration.from_pretrained(model_name)
#     self.tokenizer = T5Tokenizer.from_pretrained(model_name)
#     self.batch_size = batch_size

def forward(
    self, input_ids, attention_mask=None, labels=None #, decoder_attention_mask=None# decoder_input_ids=None, decoder_attention_mask=None
):
  return self.model(
      input_ids = input_ids,
      attention_mask=attention_mask,
      #decoder_input_ids=decoder_input_ids,
      labels=labels,
      #lm_labels=lm_labels,
      #decoder_attention_mask=decoder_attention_mask
  )

def _step(self, batch):

  outputs = self(
      input_ids=batch["source_ids"],
      attention_mask=batch["source_mask"],
      #lm_labels=lm_labels,
      labels=batch["target_ids"],
      #decoder_attention_mask=batch['target_mask']
  )

  loss = outputs.loss

  return loss

def training_step(self, batch, batch_idx):
  loss = self._step(batch)
  self.log("training_loss", loss)
  return loss

#def training_epoch_end(self, outputs):
#  avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
#  tensorboard_logs = {"avg_train_loss": avg_train_loss}
#  return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

def validation_step(self, batch, batch_idx):
  loss = self._step(batch)
  self.log("validation_loss", loss, on_epoch=True)
  return loss

#def validation_epoch_end(self, outputs):
#  avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#  tensorboard_logs = {"val_loss": avg_loss}
#  return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

def test_step(self, batch, batch_idx):
  loss = self._step(batch)

  return loss  


def configure_optimizers(self):
  "Prepare optimizer and schedule (linear warmup and decay)"

  model = self.model
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {
          "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
          "weight_decay": self.hparams.weight_decay,
      },
      {
          "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
          "weight_decay": 0.0,
      },
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
  self.opt = optimizer
  return [optimizer]

def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
  if self.trainer.use_tpu:
    xm.optimizer_step(optimizer)
  else:
    optimizer.step()
  optimizer.zero_grad()
  self.lr_scheduler.step()

def get_tqdm_dict(self):
  tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
  return tqdm_dict

def train_dataloader(self):
  dataloader = DataLoader(trainDataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
  t_total = (
      (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
      // self.hparams.gradient_accumulation_steps
      * float(self.hparams.num_train_epochs)
  )
  scheduler = get_linear_schedule_with_warmup(
      self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
  )
  self.lr_scheduler = scheduler
  return dataloader

def val_dataloader(self):
  return DataLoader(valDataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

# tokenizer = T5Tokenizer.from_pretrained("T5-base")
# model = T5FineTuner("T5-base")
# model.load_state_dict(torch.load('model/rm_base_model_v1.pt', map_location= torch.device('cpu')))
# model.eval()
MODEL_PATH = "model/t5-base-short-v1/"
device = 'cpu'
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

def WSD(sent):
    prefix = "Word Sense Disambiguation: "
    sent = prefix + sent
    encoding = tokenizer(
    [sent],
    padding="longest",
    max_length=256,
    truncation=True,
    return_tensors="pt",
    )
    
    
    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask)
    
    return tokenizer.decode(out[0])

if __name__ == '__main__':
    print(WSD("A deeply transformative memoir that can help us [ lead ] better and more fulfilling lives"))
    # Problem
    print(WSD("[ laed ]"))
    ### [O] This [ is ] a good example.
    # Note that the space between the bracket and the word is necessary, it will influence the outcome.
    ### [X] This [is] a bad example.
    doc = nlp("A deeply transformative memoir that can help us [ lead ] better and more fulfilling lives")
    print(re.search("\[.*\]", doc.text).group().strip('[]').strip(' '))
    target = re.search("\[.*\]", doc.text).group().strip('[]').strip(' ')
    for token in doc:
        if (token.text == target):
          print(token.text, token.lemma_, token.pos_)
