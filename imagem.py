import os
import shutil
from tkinter import Tk, Label, Button, Listbox
from tkinter.filedialog import askopenfilename

class ImageSelector:
    def __init__(self, master):
        self.master = master
        master.geometry("500x350")  
        master.configure(bg='navy')  
        master.title("Selecione uma imagem")

        self.listbox = Listbox(master)
        self.listbox.pack()

        
        for file in os.listdir('imagemjoao'):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.listbox.insert('end', file)

        self.select_button = Button(master, text="Selecionar", command=self.select_image)
        self.select_button.pack()

    def select_image(self):
       
        selected_image = self.listbox.get(self.listbox.curselection())

       
        if not os.path.exists('imagensaseranalizadas'):
            os.makedirs('imagensaseranalizadas')

        
        shutil.copy(os.path.join('imagemjoao', selected_image), 'imagensaseranalizadas')

       
        self.master.destroy()
class ImageMover:
    def __init__(self, master):
        self.master = master
        master.geometry("500x350")  
        master.configure(bg='navy') 
        master.title("Mover uma imagem")

        self.listbox = Listbox(master)
        self.listbox.pack()

        
        for file in os.listdir('imagensaseranalizadas'):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.listbox.insert('end', file)

        self.move_button = Button(master, text="Mover", command=self.move_image)
        self.move_button.pack()

    def move_image(self):
       
        selected_image = self.listbox.get(self.listbox.curselection())

       
        shutil.move(os.path.join('imagensaseranalizadas', selected_image), 'imagemjoao')

        
        self.master.destroy()       
