import fasttext

# Load pre-trained language identification model
model = fasttext.load_model("lid.176.bin")

def detect_language(text):
    prediction = model.predict(text)
    lang_code = prediction[0][0].replace('__label__', '')
    confidence = prediction[1][0]
    return lang_code, confidence

if __name__ == "__main__":
    text = input("Enter text to identify language: ")
    lang, conf = detect_language(text)
    print(f"Detected language: {lang} (Confidence: {conf:.2f})")
