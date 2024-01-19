from tkinter import Tk, Label, Button, Toplevel, StringVar
from tkinter.filedialog import askopenfilename
from imagem import ImageSelector
import shutil
from PIL import Image
from pytorch import Net, transform  # Importaar a classe Net e a função transform do pytorch.py
import torch
from PIL import Image
import os

class MainMenu:
    def __init__(self, master):
        self.master = master
        master.title("Menu principal")
        master.geometry("1000x700")  # Ajuste para metade do tamanho da sua tela
        master.configure(bg='navy')  # Cor de fundo azul marinho

        self.label = Label(master, text="Menu principal", bg='navy', fg='white')
        self.label.place(relx=0.5, rely=0.3, anchor='center')

        self.load_button = Button(master, text="Escolher imagem", command=self.load_image)
        self.load_button.place(relx=0.5, rely=0.4, anchor='center')

        self.put_button = Button(master, text="Colocar imagem", command=self.put_image)
        self.put_button.place(relx=0.5, rely=0.5, anchor='center')

        self.analyze_button = Button(master, text="Analisar imagem", command=self.analyze_image)
        self.analyze_button.place(relx=0.5, rely=0.6, anchor='center')

        self.image_path = None  # Variável para armazenar o caminho da imagem

    def load_image(self):
        # Abra a janela de seleção de imagem
        self.new_window = Tk()
        self.app = ImageSelector(self.new_window)
        self.new_window.mainloop()

    def put_image(self):
        # Abra a janela de seleção de arquivo
        filepath = askopenfilename(filetypes=[("Image files", "*.jpg *.png")])

        if filepath:  # Se um arquivo foi selecionado
            # Copie o arquivo para a pasta de imagens
            shutil.copy(filepath, 'imagens')
            self.image_path = filepath  # Armazene o caminho da imagem

    def analyze_image(self):
        # Abra a janela de seleção de arquivo
        filepath = askopenfilename(filetypes=[("Image files", "*.jpg *.png")])

        if filepath:  # Se um arquivo foi selecionado
            # Copie o arquivo para a pasta "analisepy"
            shutil.copy(filepath, 'analisepy')
            self.image_path = os.path.join('analisepy', os.path.basename(filepath))  # Armazene o novo caminho da imagem

            # Analise a imagem
            result = self.analyze_image_with_pytorch(self.image_path)
            print(result)  # Imprima o resultado

    def analyze_image_with_pytorch(self, filepath):
        # Carregue a imagem

        # Carregue a imagem
        image = Image.open(filepath)

        # Converta a imagem para RGB
        image = image.convert('RGB')

        # Redimensione a imagem para 32x32
        image = image.resize((32, 32))

        # Aplique as transformações
        image = transform(image)      

        # Verifique se há uma GPU disponível e, se houver, mova a imagem para a GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)

        # Carregue o modelo treinado
        model = Net()  # Inicialize a rede
        model = model.to(device)
        model.eval()

        # Passe a imagem pelo modelo
        output = model(image)

        # Obtenha o índice da classe com a maior pontuação
        _, predicted = torch.max(output, 1)

        # Mapeie o índice da classe para o nome da classe
        classes = ('cachorro', 'gato')
        result = classes[predicted.item()]

        return f"Resultado da análise: {result}"

root = Tk()
my_menu = MainMenu(root)
root.mainloop()