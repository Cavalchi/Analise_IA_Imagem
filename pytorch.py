import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((200, 200)), 
    transforms.ToTensor(),
    
])

# Definir uma CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(35344, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(trainloader):
   
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
   
    for epoch in range(50):  
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            labels = labels.long()  r
        
            optimizer.zero_grad()
           
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 2000 == 1999:  
                print('[%5d] loss: %.3f' %
                      (i + 1, running_loss / 2000))
                running_loss = 0.0
       
        torch.save(net.state_dict(), f'save/model_epoch_{epoch}.pth')
    return net  

if __name__ == "__main__":
   
    train_dataset = datasets.ImageFolder('imagensreferenca', transform=transform)    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    
    model = train_model(trainloader)

   
    torch.save(model.state_dict(), 'save/model.pth')
def analyze_image_pytorch(image_path):
    
    image = Image.open(image_path)

    
    image = image.convert('RGB')

    
    image = transform(image).unsqueeze(0)

    
    model = Net()

  
    for epoch in range(50):
        model.load_state_dict(torch.load(f'save/model_epoch_{epoch}.pth'))
        model.eval() 
        
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        
        classes = ['cachorro', 'carro', 'casa', 'gato', 'lua', 'mar', 'nuvem', 'prédio']
        predicted_class = classes[predicted.item()]

       
        print(f'Epoch {epoch + 1}, Predicted class: {predicted_class}')

    
    return predicted_class
