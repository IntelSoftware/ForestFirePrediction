aug_data=False
find_batch=False
find_lr_rate=False
use_wandb=True

use_ipex=False

set_seed(42)
print(f"Train folder {TRAIN_DIR}")
print(f"Validation folder {VALID_DIR}")
print(f"Using epoch: {EPOCHS}")
print(f"Using Dropout: {DROPOUT}")

batch_size = BATCH_SIZE

if aug_data:
    print("Augmenting training and validation datasets...")
    t1 = time.time()
    augment_and_save(TRAIN_DIR)
    augment_and_save(VALID_DIR)
    print(f"Done Augmenting in {time.time() - t1} seconds...")

model = FireFinder(simple=True, dropout=DROPOUT)
optimizer = optim.Adam(model.parameters(), lr=LR)
if find_batch:
    print(f"Finding optimum batch size...")
    batch_size = optimum_batch_size(model, input_size=(3, 1024, 1024))
print(f"Using batch size: {batch_size}")

best_lr = LR
if find_lr_rate:
    print("Finding best init lr...")
    train_dataloader = create_dataloader(
        TRAIN_DIR,
        batch_size=batch_size,
        shuffle=True,
        transform=img_transforms["train"],
    )
    best_lr = find_lr(model, optimizer, train_dataloader)
    del model, optimizer
    gc.collect()
    if device == torch.device("xpu"):
        torch.xpu.empty_cache()
print(f"Using learning rate: {best_lr}")

model = FireFinder(simple=True, dropout=DROPOUT)
trainer = Trainer(
    model=model,
    optimizer=optim.Adam,
    lr=best_lr,
    epochs=EPOCHS,
    device=device,
    use_wandb=use_wandb,
    use_ipex=use_ipex,
)
train(model, trainer, config={"lr": best_lr, "batch_size": batch_size})
