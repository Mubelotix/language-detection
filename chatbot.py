import math
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
conversation = [["Hello", "Hi"],["How are you?", "I am fine"], ["What is your name?", "My name is HAL" ]]
answer=""

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([sentence[0] for sentence in conversation])
answers=[sentence[1] for sentence in conversation]


while(True):
    answer=input(">")
    if (answer=="q"):
        break
    
    y = vectorizer.transform([answer])
    answ=[]
    for i in range(len(answers)):
        answ.append(cosine_sim(y.todense().tolist()[0], X.todense().tolist()[i]))
    print(answers[answ.index(max(answ))])
