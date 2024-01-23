# Change this in feed.py
from main import MainMenu
from main import analyze_image_with_menu
def analyze_image_feed(image_path):
    menu = MainMenu()
    result = menu.analyze_image_pytorch(image_path)
    return result