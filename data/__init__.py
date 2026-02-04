from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = "./data/datasets"
BATCH_SIZE = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
