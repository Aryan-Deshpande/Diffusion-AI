import torch
import torch.nn as nn

from torchvision import transforms

from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch.nn.functional as F

# CLIP implementation for 1 BATCH

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.5
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

"""m = ProjectionHead(768)
a1 = m(a)
print(a1.shape)"""

def cross_entropy_loss(preds, targets, reduction="none"):
  log_softmax = nn.LogSoftmax(dim=-1)
  loss = (-targets * log_softmax(preds)).sum(1)
  if reduction == "none":
    return loss
  elif reduction == "mean":
    return loss.mean()

def extract_text_embedding():

  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  encoded_query = tokenizer(["this is a text yeah"])

  batch = {
          key: torch.tensor(values)
          for key, values in encoded_query.items()
  }
  
  model = BertModel.from_pretrained('distilbert-base-uncased')

  output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
  last_hidden_state = output.last_hidden_state
  x = last_hidden_state[:, 0, :]
  print(x.shape)

  text_projection = ProjectionHead(768)
  image_projection = ProjectionHead(768)

  text_embeddings = text_projection(x)
  image_embeddings = image_projection(b)

  """image_embeddings_n = F.normalize(b, p=2, dim=-1)
  text_embeddings_n = F.normalize(x, p=2, dim=-1)"""

  logits = (text_embeddings @ image_embeddings.T) / 1.0
  texts_similarity = text_embeddings @ text_embeddings.T
  images_similarity = image_embeddings @ image_embeddings.T
  targets = F.softmax(
      (images_similarity + texts_similarity) / 2 * 1.0, dim=-1
  )

  dot_similarity = text_embeddings @ image_embeddings.T
  print(dot_similarity)

  celtext = torch.nn.CrossEntropyLoss()
  celimage = torch.nn.CrossEntropyLoss()
  image_loss = cross_entropy_loss(logits.T, targets.T, "none")
  text_loss = cross_entropy_loss(logits, targets, "none")
  print(image_loss)
  print(text_loss)

  loss = (image_loss + text_loss) / 2.0
  print(loss.mean())
  return loss.mean()

extract_text_embedding()
