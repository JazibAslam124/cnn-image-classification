import torch
import torch.nn as nn
import torch.optim as optim
from cnn_model.model import CNNModel
from cnn_model.utils import get_dataloaders
from tdqm import tdqm


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloaders()

model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 5
for epoch in range (epochs):
    model.train()
    correct, total, loss_total = 0, 0, 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion (outputs, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    print (f'Epoch{epoch+1}: Loss = {loss_total/len(train_loader):.4f} | Accuracy = {100*correct/total:.2f}%')

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
print (f"Test Accuracy: {100 * correct/total:.2f}%")



