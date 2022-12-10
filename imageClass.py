import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas
import PIL.Image
import os

num_Classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class nodes(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pandas.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = PIL.Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5], [.5])
])

train_set = nodes(annotations_file='C:/Users/mica/PycharmProjects/BloodWeb/DataSet.csv', img_dir='C:/Users/mica/PycharmProjects/BloodWeb/train', transform=transform)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

model = torchvision.models.resnet18()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_correct = 0.0
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        output_idx = torch.argmax(output, dim=1)
        total_correct += (labels == output_idx).sum().item()
        optimizer.zero_grad()
        loss = criterion(output, labels)
        running_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()

    print("Epoch:" + str(epoch) + "  loss:" + str(running_loss/train_set.__len__()) + "  Acc:" + str(total_correct/train_set.__len__()))

torch.save(model.state_dict(), "C:/Users/mica/PycharmProjects/BloodWeb/Model.pth")