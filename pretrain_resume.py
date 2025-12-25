import torch
from torch.utils.data import DataLoader

# --- import your model, dataset, and training functions ---
# Replace 'your_model_file' with the actual filename where these are defined
from your_model_file import StereoAwareGNN, pretrain_epoch, validate, dataset, head, val_loader

DEVICE = torch.device("cpu")  # change to "cuda" if you have a GPU

# --- rebuild model and optimizer ---
model = StereoAwareGNN()   # same architecture as before
optimizer = torch.optim.Adam(model.parameters(), lr=0.000523)  # same optimizer settings
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # example gamma

# --- reload checkpoint from epoch 13 ---
checkpoint = torch.load("models/checkpoints/stereo_epoch_13.pth", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1  # resume at 14

# --- safer DataLoader settings for Windows ---
train_loader = DataLoader(
    dataset,
    batch_size=256,     # your batch size
    shuffle=True,
    num_workers=0,      # avoids Windows multiprocessing hangs
    pin_memory=False    # avoids warning and stalls
)

# --- resume training loop ---
best_loss = float("inf")

for epoch in range(start_epoch, 20):
    loss = pretrain_epoch(model, head, train_loader, optimizer, DEVICE)
    val_loss = validate(model, val_loader)

    # save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, f"models/checkpoints/stereo_epoch_{epoch:02d}.pth")

    # update best model if validation improves
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "models/pretrained_stereo_full.pth")
        print(f"Epoch {epoch}: New BEST model saved with val_loss {val_loss:.6f}")