from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root='/.data', train=False, download=True, transform=transform)

    return DataLoader(train, batch_size= batch_size,shuffle=True), DataLoader(test,batch_size=batch_size)


