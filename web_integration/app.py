# app.py

from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the model (GPT-2 for text generation)
generator = pipeline("text-generation", model="gpt2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    response = generator(prompt, max_length=100, num_return_sequences=1)
    generated_text = response[0]['generated_text']
    return render_template('index.html', prompt=prompt, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
