from tkinter import Tk, Label, Button, Toplevel, StringVar, ttk, Frame, Scrollbar, Canvas
from tkinter.filedialog import askopenfilename
from imagem import ImageSelector
import shutil
from PIL import Image, ImageTk
from pytorch import Net, transform, train_model  # Importar a classe Net, a função transform e a função train_model do pytorch.py
import torch
import os
import threading
from torchvision import datasets
from pytorch import analyze_image_pytorch  # Importar a função analyze_image_pytorch do pytorch.py
class MainMenu:
    def __init__(self, master):
        self.master = master
        master.title("Menu principal")
        master.geometry("1000x700")  # Ajuste para metade do tamanho da sua tela
        master.configure(bg='navy')

        # Get the directory of the current script'
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Get the parent directory
        parent_dir = os.path.dirname(script_dir)

        # Join the script directory with the relative path to 'imagensreferenca'
        imagensreferenca_dir = os.path.join(script_dir, 'imagensreferenca') 

        # Now use 'imagensreferenca_dir' instead of 'imagensreferenca'
        train_dataset = datasets.ImageFolder(imagensreferenca_dir, transform=transform)      
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Verifique se um estado de modelo salvo existe
        if os.path.exists('save/model.pth'):
            # Carregue o estado do modelo
            self.model = Net()
            self.model.load_state_dict(torch.load('save/model.pth'))
        else:
            # Treine o modelo
            self.model = train_model(trainloader)
            # Crie o diretório 'save' se ele não existir
            if not os.path.exists('save'):
                os.makedirs('save')
            # Salve o estado do modelo
            torch.save(self.model.state_dict(), 'save/model.pth')
            # Iniciar o treinamento da rede neural em uma thread separada
            threading.Thread(target=train_model, args=(trainloader,)).start()  # Cor de fundo azul marinho
        self.label = Label(master, text="Menu principal", bg='navy', fg='white')
        self.label.place(relx=0.5, rely=0.3, anchor='center')
        self.move_image_button = Button(master, text="Analisar e mover imagem", command=self.analyze_and_move_image)
        self.move_image_button.place(relx=0.5, rely=0.8, anchor='center')
        self.explanation_button = Button(master, text="Explicação", command=self.show_explanation)
        self.explanation_button.place(relx=0.5, rely=0.7, anchor='center')

    import shutil

    def analyze_and_move_image(self):
        # Abra a janela de seleção de arquivo
        filepath = askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])

        if filepath:  # Se um arquivo foi selecionado
            # Analise a imagem
            result = analyze_image_pytorch(filepath)
            print(f"A imagem foi classificada como: {result}")
            correct = input("A classificação está correta? (s/n) ")
            if correct.lower() == 's':
                # Verifique se a pasta existe e, se não, crie a pasta
                destination_folder = f'imagensreferenca/{result}'
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                # Mova a imagem para a pasta correspondente
                shutil.move(filepath, destination_folder)
            elif correct.lower() == 'n':
                correct_class = input("Qual é a classe correta? ")
                # Verifique se a pasta existe e, se não, crie a pasta
                destination_folder = f'imagensreferenca/{correct_class}'
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                # Mova a imagem para a pasta correspondente
                shutil.move(filepath, destination_folder)
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
        filepath = askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])

        if filepath:  # Se um arquivo foi selecionado
            # Copie o arquivo para a pasta de imagens
            shutil.copy(filepath, 'imagens')
            self.image_path = filepath  # Armazene o caminho da imagem
root = Tk()
my_menu = MainMenu(root)
root.mainloop()