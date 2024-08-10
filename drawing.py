import tkinter as tk
import io
from PIL import Image, ImageOps, ImageFilter, ImageGrab
import numpy as np
from ocr import extract_features, knn_prediction


# def draw_line(event):
#     x, y = event.x, event.y
#     event.widget.create_oval((x-2, y-2, x+2, y+2), outline='black', fill='black', width=4)


DRAWING_DIR = './drawings/'


def draw_line(event):
    x, y = event.x, event.y
    if 'lastx' in canvas.data:
        canvas.create_line(canvas.data['lastx'], canvas.data['lasty'], x, y, fill='black', width=25)
    canvas.data['lastx'] = x
    canvas.data['lasty'] = y

def reset_last_position(event):
    canvas.data.pop('lastx', None)
    canvas.data.pop('lasty', None)

def close_out():

    # Save the image
    print("Saving Image")
    img_array = save_image(canvas)
    
    # Call the OCR function here


    temp_guess = knn_prediction(img_array, 5)
    print(f'Guess: {temp_guess}')

    
    # Update the guess label
    update_text(guess_label, f"OCR Guess: {temp_guess}")
    canvas.create_text(250, 20, text="Guessed!", font=("Helvetica", 20), tags="text")

    # Update the done button to say "Try Again?"
    done_button.config(text="Try Again?")
    done_button.bind("<Button-1>", lambda event: try_again_button())


def save_image(input_canvas):
    
    ps_data = input_canvas.postscript(colormode='color')
    img = Image.open(io.BytesIO(ps_data.encode('utf-8')))

    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 28x28
    img = ImageOps.invert(img)
    img = img.filter(ImageFilter.GaussianBlur(4))

    img = img.resize((28, 28), Image.ANTIALIAS)
    
    # Invert colors
    
    # Apply Gaussian blur
    
    # Save as PNG
    img.save(f'{DRAWING_DIR}temp.png')

    img_list = list(img.getdata())
    img_list = [img_list[i:i + 28] for i in range(0, len(img_list), 28)]
    binary_list = [[bytes([pixel]) for pixel in row] for row in img_list]

    wrapped_img_list = [binary_list]

    return wrapped_img_list






def try_again_button():
    canvas.delete("all")
    done_button.config(text="Done")
    done_button.bind("<Button-1>", lambda event: close_out())
    canvas.delete("all")
    update_text(guess_label, "OCR Guess: Press Done to see the result!")

def update_text(label, text):
    label.config(text=text)





def main():
    
    # Create the Window
    window = tk.Tk()
    window.title("Digit Recognizer")
    window.geometry("550x780")
    window.resizable(False, False)

    # Initialize Canvas
    global canvas
    canvas = tk.Canvas(window, width=500, height=500)
    canvas.tk_setPalette(background='white')
    canvas.config(highlightbackground="black", highlightthickness=4)
    
    # Initialize Buttons
    global done_button
    done_button = tk.Button(window, text="Done", width=50, height=5)
    clear_button = tk.Button(window, text="Clear", width=50, height=5)

    # Initialize Labels
    canvas_label = tk.Label(window, text="Draw a digit")
    done_button_label = tk.Label(window, text="Click done when you're done drawing")
    clear_button_label = tk.Label(window, text="Click clear to clear the canvas")
    global guess_label
    guess_label = tk.Label(window, text="OCR Guess: Press Done to see the result!")

    # Pack Labels
    canvas_label.pack()
    done_button_label.pack()
    clear_button_label.pack()
    
    
    # Pack Widgets
    canvas.pack()
    guess_label.pack()
    done_button.pack()
    clear_button.pack()

    canvas.bind('<B1-Motion>', draw_line)
    canvas.bind('<ButtonRelease-1>', reset_last_position)
    canvas.data = {}


    clear_button.bind("<Button-1>", lambda event: canvas.delete("all"))
    done_button.bind("<Button-1>", lambda event: close_out())






    window.mainloop()







if __name__ == '__main__':
    main()