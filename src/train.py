
import torch
import tqdm
from utils.common_functions import AverageMeter

from config.model_config import train, dataloaders


# -----------------------------------------------------------------------------------------------
# ------------------------------------ TRAIN ----------------------------------------------------
# -----------------------------------------------------------------------------------------------
def train_one_epoch(model, train_loader, loss_fn, optimizer, metric=None, epoch=None, device=train['device']):
  
  model.train()
  loss_train = AverageMeter()
  if metric:
    metric.reset()

  with tqdm.tqdm(train_loader, unit='batch') as tepoch:
    for inputs, targets in tepoch:
      if epoch:
        tepoch.set_description(f'Epoch {epoch}')

      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs, targets)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

      loss_train.update(loss.item(), n=len(targets))
      if metric:
        metric.update(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())


      tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item() if metric else None)

  return model, loss_train.avg, metric.compute().item() if metric else None




# -----------------------------------------------------------------------------------------------
# ------------------------------------ EVALUATION -----------------------------------------------
# -----------------------------------------------------------------------------------------------
def evaluate(model, test_loader, loss_fn, metric=None, device=train['device']):
  model.eval()
  loss_eval = AverageMeter()
  if metric:
    metric.reset()

  with torch.inference_mode():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      targets = targets.to(device)

      outputs = model(inputs, targets)

      loss = loss_fn(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())
      loss_eval.update(loss.item(), n=len(targets))

      if metric:
        metric.update(outputs.reshape(-1, outputs.shape[-1]), targets.flatten())

      return loss_eval.avg, metric.compute().item() if metric else None