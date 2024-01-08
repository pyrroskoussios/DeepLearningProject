import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from pathlib import Path

from training_utils import load_dataset, VGG, ResNet

def train_model(datas, model, optimizer, criterion, scheduler, epochs, device, checkpoint_path, save_epoch = 1):
	model.to(device=device)
	train_loader, test_loader = datas
	start_time = time.time()
	best_val_loss = 999999
	for epoch in range(1, epochs + 1):
		train_loss = 0
		# Train for one epoch
		model.train()
		with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
			for i, batch in enumerate(train_loader):
				# Forward pass
				x, ygt = batch
			   
				x = x.to(device=device)
				ygt = ygt.to(device=device)
				ypr = model(x)
				loss = criterion(ypr, ygt)

				# Backward pass
				optimizer.zero_grad()
				loss.backward()

				#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

				optimizer.step()
				train_loss += loss.item()

				# Update the learning rate if using a scheduler

				pbar.update(x.shape[0])
				pbar.set_postfix(**{'acc (batch)': 100*((torch.argmax(ypr, 1) == ygt).sum().item())/ygt.size(0)  })
			scheduler.step()

		with torch.no_grad():
			model.eval()
			correct = 0
			total = 0
			val_loss = 0.0
			for j, batch in enumerate(test_loader):
				x, ygt = batch
				ypr = model(x)
				loss = criterion(ypr, ygt)
				val_loss += loss.item()
				predicted = torch.argmax(ypr, 1)
				total += ygt.size(0)
				correct += (predicted == ygt).sum().item()
			print(f'Epoch {epoch} complete. Validation loss: {val_loss / 5000:.3f}, '+
			  f'validation accuracy: {100 * correct / total}')

			if val_loss < best_val_loss:
				best_state_dict = model.state_dict()
				best_val_loss = val_loss

		# Save model checkpoint
		if epoch % save_epoch == 0:
			Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
			state_dict = model.state_dict()
			torch.save(state_dict, os.path.join(checkpoint_path, 'checkpoint_epoch{}.pth'.format(epoch)))

	torch.save(best_state_dict, os.path.join(checkpoint_path, 'best_val_epoch.pth'))


	print('Finished training in', round(time.time()-start_time,3),'seconds')
	return model

def initialisation():
	if not os.path.exists(os.path.join(os.getcwd(), 'checkpoints')):
		os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
	checkpoint_path = os.path.join(os.getcwd(), 'checkpoints')
	print("GPU available: ", torch.cuda.is_available())
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.backends.cuda.matmul.allow_tf32 = True
	torch.set_default_dtype(torch.float32)

	return checkpoint_path, device

if __name__ == "__main__":
	checkpoint_path, device = initialisation()
	torch.manual_seed(42)

	model_type = "VGG"
	dataset_type = 'CIFAR10'
	class_num = 100 if dataset_type == 'CIFAR100' else 10
	batch_size = 64
	epochs = 30
	lr = 0.05

	if model_type == "VGG":
		model = VGG('VGG11', class_num, batch_norm = False, bias = False)
	else:
		model = ResNet(BasicBlock, [2,2,2,2], num_classes=class_num, use_batchnorm=False, linear_bias=False)
	
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
	loss_function = nn.CrossEntropyLoss()
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 20, 25, 28], gamma=0.1)
	data = load_dataset(dataset_type, batch_size)
	
	print("Train batch count", len(data[0]), ", train img count", len(data[0].dataset))
	print("Test batch count", len(data[1]), ", test img count", len(data[1].dataset))
	print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")

	torch.save(model.state_dict(), os.path.join(checkpoint_path, 'initial_weights.pth'))
	model = train_model(data, model, optimizer, loss_function, scheduler, epochs, device, checkpoint_path)









