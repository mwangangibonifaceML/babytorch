import numpy as np

from minitorch.tensor.tensor import Tensor
from minitorch.dataloaders.dataloader import DataLoader, TensorDataset
from minitorch.zexamples.loan_default.src.components.data_transformation import DataTransformation
from minitorch.zexamples.loan_default.src.pipeline.model_pipeline import LoanDefaultPredictor
from minitorch.losses.losses import BinaryCrossEntropy
from minitorch.optimizers.optim import SGD, Adam, AdamW
from minitorch.train.training import Trainer, CosineSchedule

MAX_EPOCHS = 200
STEPS = MAX_EPOCHS / 10

# Ingestion and transform the data
data_transformer = DataTransformation()
train_arr, test_arr = data_transformer.initiate_data_transformation()

# get the features and target and convert them to tensors
features, target = train_arr[:, 1:-2], train_arr[:, -1]
features, target = Tensor(features, requires_grad=True), Tensor(target, requires_grad=True)

# create the data loader
ds = TensorDataset(features, target)
train_loader = DataLoader(ds, batch_size=32, shuffle=False)

# create the model
model = LoanDefaultPredictor(in_features=features.shape[1], out_features=1, drop_out_p=0.0)
loss_fn = BinaryCrossEntropy()
# optimizer = SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.1)
optimizer = AdamW(model.parameters(), lr=0.03, weight_decay=0.1)
scheduler = CosineSchedule(max_lr=0.03, min_lr = 0.003, total_epochs=MAX_EPOCHS)
trainer = Trainer(model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                clip_gradients=True)

print()
print('Full model training...\n')
for epoch in range(MAX_EPOCHS):
    loss = trainer.train_epoch(train_loader)
        
    if (epoch + 1) % STEPS == 0:
        print(f'Epoch {epoch+1}/{MAX_EPOCHS} | Average Loss: {loss:.4f}')
        print(f'Learning Rate at epoch {epoch+1}: {trainer.scheduler.get_lr(epoch):.6f}\n')
        
        for p in model.parameters():
            print(f'Parameter: {p.data.flatten()[:5]} | Grad: {p.grad.flatten()[:5]}')
        print('\n')
    
    
    

