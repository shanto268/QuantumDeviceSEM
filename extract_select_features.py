import PySimpleGUI as sg
from utils import *

def main():
    # Define the layout
    layout = [
        [sg.Text("SEM Image Analyzer")],
        [sg.Text("Enter the path to the image:"), sg.Input(key="-IMAGE-"), sg.FileBrowse()],
        [sg.Text("Enter the pixel scale (default is 370):"), sg.Input(default_text="370", key="-SCALE-")],
        [sg.Text("Enter the number of clicks needed to extract the feature:"), sg.Input(key="-INPUTS-")],
        [sg.Button("Submit"), sg.Button("Exit")]
    ]

    # Create the window
    window = sg.Window("SEM Image Analyzer", layout)

    # Event loop
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Submit":
            image_path = values["-IMAGE-"]
            pixel_scale = int(values["-SCALE-"])
            num_input = int(values["-INPUTS-"])
            process_image(image_path, pixel_scale, num_input)

    window.close()

if __name__ == "__main__":
    main()

