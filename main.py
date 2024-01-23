from tkinter import Tk, Label, Button, Toplevel, StringVar
from tkinter.filedialog import askopenfilename
from imagem import ImageSelector
import shutil
from PIL import Image
from pytorch import Net, transform, train_model  # Importar a classe Net, a função transform e a função train_model do pytorch.py
import torch
from PIL import Image
import os
import threading
from feed import analyze_image_feed
from torchvision import datasets
class MainMenu:
    def __init__(self, master):
        self.master = master
        master.title("Menu principal")
        master.geometry("1000x700")  # Ajuste para metade do tamanho da sua tela
        master.configure(bg='navy')
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Join the script directory with the relative path to 'imagensreferenca'
        imagensreferenca_dir = os.path.join(script_dir, 'imagensreferenca')

            # Now use 'imagensreferenca_dir' instead of 'imagensreferenca'
        train_dataset = datasets.ImageFolder(imagensreferenca_dir, transform=transform)      
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Iniciar o treinamento da rede neural em uma thread separada
        threading.Thread(target=train_model, args=(trainloader,)).start()  # Cor de fundo azul marinho

        self.label = Label(master, text="Menu principal", bg='navy', fg='white')
        self.label.place(relx=0.5, rely=0.3, anchor='center')

        self.load_button = Button(master, text="Escolher imagem", command=self.load_image)
        self.load_button.place(relx=0.5, rely=0.4, anchor='center')

        self.put_button = Button(master, text="Colocar imagem", command=self.put_image)
        self.put_button.place(relx=0.5, rely=0.5, anchor='center')

        self.analyze_button = Button(master, text="Analisar imagem", command=self.analyze_image)
        self.analyze_button.place(relx=0.5, rely=0.6, anchor='center')

        self.explanation_button = Button(master, text="Explicação", command=self.show_explanation)
        self.explanation_button.place(relx=0.5, rely=0.7, anchor='center')
    
        self.image_path = None  # Variável para armazenar o caminho da imagem


    def show_explanation(self):
        explanation_window = Toplevel(self.master)
        explanation_window.title("Explicação")

        # Crie um frame para conter a visualização da árvore e as imagens
        frame = Frame(explanation_window)
        frame.pack()

        # Crie a visualização da árvore
        tree = ttk.Treeview(frame)
        tree.pack(side="left")

        # Crie um canvas para exibir as imagens
        canvas = Canvas(frame)
        canvas.pack(side="right")

        # Adicione uma barra de rolagem ao canvas
        scrollbar = Scrollbar(frame, command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        # Configure o canvas para usar a barra de rolagem
        canvas.configure(yscrollcommand=scrollbar.set)

        # Crie um frame para conter as imagens
        image_frame = Frame(canvas)
        canvas.create_window((0,0), window=image_frame, anchor="nw")

        def populate_tree(parent_node, path):
            for p in os.scandir(path):
                id = tree.insert(parent_node, "end", text=p.name, values=[p.path])
                if not p.is_file():
                    # it might have subdirectory
                    tree.insert(id, "end")

        def update_tree(event):
            tree = event.widget
            selected_item = tree.focus()
            path = tree.item(selected_item)['values'][0]
            populate_tree(selected_item, path)

        def show_images(event):
            # Clear the image frame
            for widget in image_frame.winfo_children():
                widget.destroy()

            tree = event.widget
            selected_item = tree.focus()
            path = tree.item(selected_item)['values'][0]

            if os.path.isdir(path):
                image_files = os.listdir(path)

                for image_file in image_files:
                    image_path = os.path.join(path, image_file)
                    image = Image.open(image_path)
                    photo = ImageTk.PhotoImage(image)

                    # Add the image to the frame
                    label = Label(image_frame, image=photo)
                    label.image = photo  # Keep a reference to the image
                    label.pack()

            # Update the scrollregion of the canvas after adding all the images
            explanation_window.update()
            canvas.configure(scrollregion=canvas.bbox("all"))

        abspath = os.path.abspath('analisepy')  # Change this to the root directory
        root_node = tree.insert('', 'end', text=abspath, values=[abspath], open=True)
        populate_tree(root_node, abspath)
        tree.bind('<<TreeviewOpen>>', update_tree)
        tree.bind('<<TreeviewSelect>>', show_images)
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
        if self.image_path:
            result = analyze_image_feed(self, self.image_path)

root = Tk()
my_menu = MainMenu(root)
root.mainloop()