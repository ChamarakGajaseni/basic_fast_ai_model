import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from fastai.vision.all import *
import gradio as gr

def is_pitbull(x): return 'american_pit_bull' in x.lower()

learn = load_learner('model.pkl')
pathlib.PosixPath = temp

categories = ('not a Pitbull','a Pitbull')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))


image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()


examples = ['images\pitbull.jpg','images\cat.jpg','images\dog.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch()