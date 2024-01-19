import os
import shutil
from tkinter import Tk, Label, Button, Listbox
from tkinter.filedialog import askopenfilename

class ImageSelector:
    def __init__(self, master):
        self.master = master
        master.geometry("500x350")  # Ajuste para metade do tamanho da sua tela
        master.configure(bg='navy')  # Cor de fundo azul marinho
        master.title("Selecione uma imagem")

        self.listbox = Listbox(master)
        self.listbox.pack()

        # Lista as imagens na pasta "imagensjoao"
        for file in os.listdir('imagemjoao'):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.listbox.insert('end', file)

        self.select_button = Button(master, text="Selecionar", command=self.select_image)
        self.select_button.pack()

    def select_image(self):
        # Obtém a imagem selecionada
        selected_image = self.listbox.get(self.listbox.curselection())

        # Cria a pasta "imagensaseranalizadas" se ela não existir
        if not os.path.exists('imagensaseranalizadas'):
            os.makedirs('imagensaseranalizadas')

        # Copia a imagem para a pasta "imagensaseranalizadas"
        shutil.copy(os.path.join('imagemjoao', selected_image), 'imagensaseranalizadas')

        # Fecha a janela atual
        self.master.destroy()
class ImageMover:
    def __init__(self, master):
        self.master = master
        master.geometry("500x350")  # Ajuste para metade do tamanho da sua tela
        master.configure(bg='navy')  # Cor de fundo azul marinho
        master.title("Mover uma imagem")

        self.listbox = Listbox(master)
        self.listbox.pack()

        # Lista as imagens na pasta "imagensaseranalizadas"
        for file in os.listdir('imagensaseranalizadas'):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.listbox.insert('end', file)

        self.move_button = Button(master, text="Mover", command=self.move_image)
        self.move_button.pack()

    def move_image(self):
        # Obtém a imagem selecionada
        selected_image = self.listbox.get(self.listbox.curselection())

        # Move a imagem para a pasta "imagemjoao"
        shutil.move(os.path.join('imagensaseranalizadas', selected_image), 'imagemjoao')

        # Fecha a janela atual
        self.master.destroy()       