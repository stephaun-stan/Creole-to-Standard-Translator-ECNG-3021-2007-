from transformers import BartForConditionalGeneration, BartTokenizer
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

#creates Flask api
app = Flask(__name__)
CORS(app, resources={r"/translate": {"origins": "*"}})  # Adds CORS middle ware for flask app


# Loads the trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("imchris/BART-creole-to-english")
tokenizer = BartTokenizer.from_pretrained("imchris/BART-creole-to-english")


# Function to generate translations using the BART model
def translate_creole(creole_text, max_length=128):
    try:
        if not creole_text.strip():
           return "Input sentence is empty."
        # Tokenize and encode the input sentence
        input_ids = tokenizer.encode(creole_text, return_tensors='pt', truncation=True, max_length=max_length)

        # Ensures that the input is not empty
        if input_ids.numel() == 0:
            return "Input sentence is empty or too long."
        
        if len(creole_text) > (max_length+20):
            raise ValueError(f"Input sentence length ({len(creole_text)}) exceeds the maximum allowed length.")


        # Generate translation
        output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, length_penalty=0.6, no_repeat_ngram_size=2)
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        return f"Error during translation: {str(e)}"

# Defines the route for rendering the html template
@app.route('/')
def home():
    return render_template('index.html')

# Defines the /translate route for handling translation requests
@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Get the input text from the request
        data = request.get_json()
        input_text = data.get('text')

        # Uses pre-trained BART model for translation here
        translated_text = translate_creole(input_text)
        
        # Return the translated text as JSON
        return jsonify({'translatedText': translated_text, 'message': 'Model loaded successfully'})
    
    except ValueError as ve:
        return jsonify({'error': str(ve), 'message': 'Input sentence length exceeds max_length'})

    
    # Return an error message if there's an exception
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
