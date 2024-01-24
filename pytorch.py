import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# Transformações a serem aplicadas nas imagens
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # Redimensiona todas as imagens para 200x200
    transforms.ToTensor(),
    # Adicione quaisquer outras transformações que você esteja usando aqui
])

# Definir uma CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(35344, 120)  # Alterado para 35344
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)  # Alterado para 8 para refletir o número correto de classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Ajustar o tamanho do tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(trainloader):
    # Inicializar o modelo e o otimizador
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Loop de treinamento
    for epoch in range(50):  # Treinar por 50 épocas
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Obter as entradas
            inputs, labels = data
            labels = labels.long()  # Converter os rótulos para LongTensor
            # Zerar os gradientes do otimizador
            optimizer.zero_grad()
            # Forward + backward + otimização
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Imprimir estatísticas de perda
            running_loss += loss.item()
            if i % 2000 == 1999:  # Imprimir a cada 2000 mini-lotes
                print('[%5d] loss: %.3f' %
                      (i + 1, running_loss / 2000))
                running_loss = 0.0
        # Salve o estado do modelo após cada época
        torch.save(net.state_dict(), f'save/model_epoch_{epoch}.pth')
    return net  # Retorne o modelo treinado

if __name__ == "__main__":
    # Carregar o conjunto de dados de treinamento
    train_dataset = datasets.ImageFolder('imagensreferenca', transform=transform)    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Treine o modelo
    model = train_model(trainloader)

    # Salve o estado do modelo
    torch.save(model.state_dict(), 'save/model.pth')
def analyze_image_pytorch(image_path):
    # Carregar a imagem
    image = Image.open(image_path)

    # Convert the image from RGBA to RGB
    image = image.convert('RGB')

    # Aplicar as transformações e adicionar uma dimensão extra
    image = transform(image).unsqueeze(0)

    # Carregar o modelo treinado
    model = Net()

    # Loop sobre os estados salvos do modelo
    for epoch in range(50):
        model.load_state_dict(torch.load(f'save/model_epoch_{epoch}.pth'))
        model.eval()  # Colocar o modelo em modo de avaliação

        # Analisar a imagem
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        # Mapear o índice previsto para a classe correspondente
        classes = ['cachorro', 'carro', 'casa', 'gato', 'lua', 'mar', 'nuvem', 'prédio']
        predicted_class = classes[predicted.item()]

        # Imprimir a classe prevista para cada época
        print(f'Epoch {epoch + 1}, Predicted class: {predicted_class}')

    # Retornar a classe prevista na última época
    return predicted_class