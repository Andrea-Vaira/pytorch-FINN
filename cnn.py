import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Check if GPU is available
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

# --- 2. COSTRUZIONE DELLA CNN ---
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # In PyTorch, shapes are: (batch_size, channels, height, width)
        # CIFAR-10 starts with 3 channels (RGB)
        
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3) # Output: (32, 30, 30)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                     # Output: (32, 15, 15)
        
        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3) # Output: (64, 13, 13)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                     # Output: (64, 6, 6)
        
        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3) # Output: (64, 4, 4)
        
        # Livelli densi (Dense/Linear layers)
        # 64 channels * 4 * 4 feature map size = 1024 features after flattening
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10) # 10 nodi di uscita (logits)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        
        # Flattening
        x = x.view(-1, 64 * 4 * 4) 
        
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) # Output logits (from_logits=True equivalent)
        return x

model = CIFAR10CNN().to(device)
print(model)

# --- 3. COMPILAZIONE E ALLENAMENTO ---
# Loss equivalent to SparseCategoricalCrossentropy(from_logits=True)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# History dictionaries to store metrics for plotting
history = {
    'accuracy': [], 'val_accuracy': [],
    'loss': [], 'val_loss': []
}

epochs = 10

for epoch in range(epochs):
    # Training Phase
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
        
    epoch_loss = running_loss / len(trainloader.dataset)
    epoch_acc = correct_train / total_train

    # Validation Phase
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

# --- 4. VALUTAZIONE E PERFORMANCE ---
print(f"\n🎯 Accuracy finale sul Test Set: {history['val_accuracy'][-1]:.4f}")

# --- 5. GRAFICO DELLE PERFORMANCE ---
plt.figure(figsize=(10, 4))

# Grafico della precisione (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label = 'Test Accuracy')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Andamento Accuracy')

# Grafico della perdita (Loss)
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label = 'Test Loss')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Andamento Perdita')

plt.savefig('/home/andrea/Desktop/Tensorflow tutorial/loss_curve.png', dpi=300, bbox_inches='tight')
print("Plot saved successfully!")