# feed.py
from main import analyze_image_with_menu

def analyze_image_feed(menu, image_path):
    result = menu.analyze_image_pytorch(image_path)
    return result