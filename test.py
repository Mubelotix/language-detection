import json
import fasttext

languages = ["en", "es", "el", "ja"]

model = fasttext.load_model("lid.176.bin")

def detect_language(text):
    prediction = model.predict(text)
    lang_code = prediction[0][0].replace('__label__', '')
    confidence = prediction[1][0]
    return lang_code, confidence

for language in languages:
    file = open(language + "_1000.jsonl", "r")
    lines = file.readlines()
    errors = 0
    confidence_sum = 0
    lowest_confidence = 999
    lowest_confidence_text = ""
    for line in lines:
        parsed = json.loads(line.strip())
        text = parsed["text"].replace("\n", " ").replace("\r", " ")
        lang, conf = detect_language(text)
        if lang != language:
            errors += 1
            print(f"Detected language: {lang} instead of {language} (Confidence: {conf:.2f})")
        else:
            confidence_sum += conf
            if conf < lowest_confidence:
                lowest_confidence = conf
                lowest_confidence_text = text
    file.close()
    print(f"Errors for {language}: {errors}")
    print(f"Average confidence for {language}: {confidence_sum / (len(lines) - errors):.2f}")
    print(f"Lowest confidence for {language}: {lowest_confidence:.2f} for text: {lowest_confidence_text}")
