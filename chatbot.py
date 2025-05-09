import math
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sim(vec1, vec2):
    """ Let's convert our dictionaries to lists for easier matching."""
    vec1 = [val for val in vec1]
    vec2 = [val for val in vec2]
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))
    return dot_prod / (mag_1 * mag_2)

answer=""
conversations = {
    "en": [["Hello", "Hi"],["How are you?", "I am fine"], ["What is your name?", "My name is HAL" ]],
    "fr": [["Bonjour", "Salut"],["Comment ça va?", "Je vais bien"], ["Quel est ton nom?", "Mon nom est HAL" ]],
    "de": [["Hallo", "Hallo"],["Wie geht es dir?", "Mir geht es gut"], ["Wie heißt du?", "Mein Name ist HAL" ]],
    "es": [["Hola", "Hola"],["¿Cómo estás?", "Estoy bien"], ["¿Cuál es tu nombre?", "Mi nombre es HAL" ]],
    "it": [["Ciao", "Ciao"],["Come stai?", "Sto bene"], ["Qual è il tuo nome?", "Il mio nome è HAL" ]],
    "pt": [["Olá", "Oi"],["Como você está?", "Estou bem"], ["Qual é o seu nome?", "Meu nome é HAL" ]],
    "pl": [["Cześć", "Cześć"],["Jak się masz?", "Mam się dobrze"], ["Jak masz na imię?", "Mam na imię HAL" ]],
    "cs": [["Ahoj", "Ahoj"],["Jak se máš?", "Mám se dobře"], ["Jak se jmenuješ?", "Jmenuji se HAL" ]],
    "ja": [["こんにちは", "こんにちは"],["お元気ですか？", "元気です"], ["あなたの名前は何ですか？", "私の名前はHALです" ]],
}

vectorizers = {}
for lang, conv in conversations.items():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([sentence[0] for sentence in conv])
    answers=[sentence[1] for sentence in conv]
    vectorizers[lang] = (vectorizer, X, answers)

# Load pre-trained language identification model
model = fasttext.load_model("lid.176.bin")

def detect_language(text):
    prediction = model.predict(text)
    lang_code = prediction[0][0].replace('__label__', '')
    confidence = prediction[1][0]
    return lang_code, confidence

while(True):
    answer=input(">")
    if (answer=="q"):
        break

    lang, conf = detect_language(answer)

    if not lang in vectorizers:
        print("Language", lang, "not supported")
        continue

    vectorizer, X, answers = vectorizers[lang]
    
    y = vectorizer.transform([answer])
    answ=[]
    for i in range(len(answers)):
        try:
            answ.append(cosine_sim(y.todense().tolist()[0], X.todense().tolist()[i]))
        except:
            answ.append(0)
    print(answers[answ.index(max(answ))])

