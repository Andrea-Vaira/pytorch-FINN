import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import brevitas.nn as qnn
from brevitas.export import export_qonnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. CARICAMENTO E PREPARAZIONE DATI ---
# In PyTorch, data normalization is handled via transforms.
# transforms.ToTensor() automatically scales pixels from [0, 255] to [0.0, 1.0]
transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

#define model

class CIFAR10QuantCNN(nn.Module):
    def __init__(self, weight_bits=4, act_bits=4):
        super(CIFAR10QuantCNN, self).__init__()

        # Sostituiamo nn.Conv2d con QuantConv2d e specifichiamo i bit
        self.conv1 = qnn.QuantConv2d(in_channels=3, out_channels=32, kernel_size=3, weight_bit_width=weight_bits)
        self.relu1 = qnn.QuantReLU(bit_width=act_bits)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = qnn.QuantConv2d(in_channels=32, out_channels=64, kernel_size=3, weight_bit_width=weight_bits)
        self.relu2 = qnn.QuantReLU(bit_width=act_bits)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = qnn.QuantConv2d(in_channels=64, out_channels=64, kernel_size=3, weight_bit_width=weight_bits)
        self.relu3 = qnn.QuantReLU(bit_width=act_bits)
        
        # Flattening verrà fatto nel forward
        
        self.fc1 = qnn.QuantLinear(64 * 4 * 4, 64, weight_bit_width=weight_bits)
        self.relu4 = qnn.QuantReLU(bit_width=act_bits)
        
        # L'ultimo layer (logits) spesso si tiene con più precisione o non quantizzato per preservare l'accuracy
        self.fc2 = qnn.QuantLinear(64, 10, weight_bit_width=8)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        
        x = x.view(-1, 64 * 4 * 4) 
        
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x
    
    

model = CIFAR10QuantCNN().to(device)
#print(model)

# training

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = {
    'accuracy': [], 'val_accuracy': [],
    'loss': [], 'val_loss': []
}

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()       # Reset gradients
        outputs = model(inputs)     # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()             # Backward pass
        optimizer.step()            # Update weights
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss/len(trainloader.dataset)
    epoch_acc = correct_train/total_train

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # Disable gradient calculation for efficiency
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    epoch_val_loss = val_loss / len(testloader.dataset)
    epoch_val_acc = correct_val / total_val
    
    # Save historical data
    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

print(f"\n🎯 Accuracy finale sul Test Set: {history['val_accuracy'][-1]:.4f}")