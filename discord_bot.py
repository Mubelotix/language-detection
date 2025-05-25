import discord
import math
import fasttext
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Cosine similarity function
def cosine_sim(vec1, vec2):
    dot_prod = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    mag_1 = math.sqrt(sum(x**2 for x in vec1))
    mag_2 = math.sqrt(sum(x**2 for x in vec2))
    return dot_prod / (mag_1 * mag_2 + 1e-9)

# Sample conversations
conversations = {
    "en": [
        ["Hello", "Hi there!"],
        ["How are you?", "I'm doing great, thank you!"],
        ["What is your name?", "My name is HAL."],
        ["What's your name?", "I'm HAL, your friendly assistant."],
        ["Who are you?", "I'm HAL, your virtual assistant."],
        ["Good morning", "Good morning to you too!"],
        ["Good night", "Sleep well!"],
        ["What's the weather like?", "I can't check the weather, but I hope it's nice!"],
        ["What time is it?", "I don't have a clock, but time flies when you're chatting!"],
        ["What day is it?", "Let me guess... it's today!"],
        ["Tell me a joke", "Why did the computer go to therapy? It had too many bytes."],
        ["Thank you", "You're very welcome!"],
        ["Thanks", "Anytime!"],
        ["Bye", "Goodbye! Talk to you soon."],
        ["See you later", "Looking forward to it!"],
        ["Where are you from?", "I'm from the cloud."],
        ["How old are you?", "Old enough to be wise, young enough to keep learning."],
        ["What do you do?", "I help people with information and conversation."],
        ["Help me", "Sure! How can I assist you?"],
        ["I need assistance", "I'm here to help. What's the issue?"],
        ["You're funny", "Glad you think so!"],
        ["You're stupid", "That's not very nice 😢"],
        ["Do you like me?", "Of course! I enjoy talking to you."],
    ],
    "fr": [["Bonjour", "Salut"], ["Comment ça va?", "Je vais bien"], ["Quel est ton nom?", "Mon nom est HAL"]],
    "de": [["Hallo", "Hallo"], ["Wie geht es dir?", "Mir geht es gut"], ["Wie heißt du?", "Mein Name ist HAL"]],
    "es": [["Hola", "Hola"], ["¿Cómo estás?", "Estoy bien"], ["¿Cuál es tu nombre?", "Mi nombre es HAL"]],
    "it": [["Ciao", "Ciao"], ["Come stai?", "Sto bene"], ["Qual è il tuo nome?", "Il mio nome è HAL"]],
    "pt": [["Olá", "Oi"], ["Como você está?", "Estou bem"], ["Qual é o seu nome?", "Meu nome é HAL"]],
    "pl": [
        ["Cześć", "Cześć!"],
        ["Jak się masz?", "Mam się świetnie, dziękuję!"],
        ["Jak masz na imię?", "Mam na imię HAL."],
        ["Kim jesteś?", "Jestem HAL, Twój wirtualny asystent."],
        ["Dzień dobry", "Dzień dobry również!"],
        ["Dobranoc", "Śpij dobrze!"],
        ["Jaka jest pogoda?", "Nie mogę sprawdzić pogody, ale mam nadzieję, że jest ładna!"],
        ["Która jest godzina?", "Nie mam zegarka, ale czas leci podczas rozmowy!"],
        ["Jaki jest dzisiaj dzień?", "Zgaduję... dzisiaj!"],
        ["Opowiedz mi dowcip", "Dlaczego komputer poszedł na terapię? Bo miał zbyt wiele bajtów."],
        ["Dziękuję", "Bardzo proszę!"],
        ["Dzięki", "Zawsze do usług!"],
        ["Do widzenia", "Do zobaczenia!"],
        ["Do zobaczenia później", "Nie mogę się doczekać!"],
        ["Skąd jesteś?", "Jestem z chmury."],
        ["Ile masz lat?", "Wystarczająco dużo, by być mądrym, ale młodym duchem."],
        ["Co robisz?", "Pomagam ludziom, udzielając informacji i prowadząc rozmowy."],
        ["Pomóż mi", "Oczywiście! W czym mogę pomóc?"],
        ["Potrzebuję pomocy", "Jestem tutaj, aby pomóc. Jaki masz problem?"],
        ["Jesteś zabawny", "Cieszę się, że tak myślisz!"],
        ["Jesteś głupi", "To nie było miłe 😢"],
        ["Lubisz mnie?", "Oczywiście! Lubię z Tobą rozmawiać."],
    ],
    "cs": [["Ahoj", "Ahoj"], ["Jak se máš?", "Mám se dobře"], ["Jak se jmenuješ?", "Jmenuji se HAL"]],
    "ja": [["こんにちは", "こんにちは"], ["お元気ですか？", "元気です"], ["あなたの名前は何ですか？", "私の名前はHALです"]],
}

# Preload TF-IDF models
vectorizers = {}
for lang, conv in conversations.items():
    vectorizer = TfidfVectorizer()
    questions = [pair[0] for pair in conv]
    answers = [pair[1] for pair in conv]
    X = vectorizer.fit_transform(questions)
    vectorizers[lang] = (vectorizer, X, answers)

# Load FastText language detection model
model = fasttext.load_model("lid.176.bin")

def detect_language(text):
    prediction = model.predict(text)
    lang_code = prediction[0][0].replace("__label__", "")
    return lang_code

def get_response(text):
    lang = detect_language(text)
    if lang not in vectorizers:
        return f"Language '{lang}' not supported."
    
    print(f"Detected language: {text}")

    vectorizer, X, answers = vectorizers[lang]
    y = vectorizer.transform([text])
    y_vec = y.todense().tolist()[0]
    sims = [cosine_sim(y_vec, X[i].todense().tolist()[0]) for i in range(len(answers))]
    best_index = sims.index(max(sims))
    return answers[best_index]

# Initialize Discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"HAL is online as {client.user}")

@client.event
async def on_message(message):
    print(message)
    if message.author == client.user:
        return
    response = get_response(message.content)
    await message.channel.send(response)

token = os.environ['DISCORD_TOKEN']
client.run(token)
