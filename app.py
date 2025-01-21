import streamlit as st
from dotenv import load_dotenv
import json
import os
import pandas as pd
import time
import random
from pycaret.regression import load_model, predict_model
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import OpenAI as LangfuseOpenAI

# Wczytanie zmiennych z .env
load_dotenv()

# Wczytanie modelu regresyjnego
model = load_model('hm_model')

# Ustawienie początkowej wartości w session_state
if 'text' not in st.session_state:
    st.session_state.text = ""

# Inicjacja Langfuse
langfuse = Langfuse()
langfuse.auth_check()
llm_client = LangfuseOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Zapytanie do openAI, podpięte pod Langfuse   
@observe()
def get_data_from_text(text,model="gpt-4o-mini"):

    prompt = """
    You are an amazing help in finding information in text.
    You will be provided with short text that contains the following data about runner:

    <sex> - choose whether the runner is a man (M) or a woman (F).
    <age> - determine the age of the runner in years.
    <time> - specify the time in minutes for the runner to cover a distance of 5 kilometers, if it provides a range, choose a lower value

    Return the value as a dictionary the following keys:
    - sex - as a string "F" or "M" or null if not provided
    - age - as an integer or null if not provided
    - time - as an integer or null if not provided

    The text will be provided in polish and you should respond with the json as defined above
    and you should use exactly those keys. Return a valid dictionary, nothing else.
    I will parse the result with json.loads() function in python so please make sure the result is valid.
   
    Here is an example of the text content:
    ```
    Jestem mężczyzną i mam 32 lata mój najlepszy czas na 5 km to 27 minut
    ```

    In this case the dictionary should look like this:
    {
    "sex": "M",
    "age": 32,
    "time": 27
    }
    """

    messages=[
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"```{text}```",
        },
    ]

    chat_completion = llm_client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=messages, 
        model=model,
        # dodatkowe
        name="get_data_from_text",
    )
    resp = chat_completion.choices[0].message.content
    try:
        output = json.loads(resp)
    except:
        output = {"error": resp}
    return output
    
sentences = ['Biegaj nie tylko po to, by dotrzeć do celu, ale by cieszyć się każdą chwilą',
             'Biegaj z sercem, a nie tylko z nogami',
             'Każdy krok to postęp, nawet jeśli czasem upadniesz',
             'Biegnij nie po to, by wygrać, ale by pokonać samego siebie',
             'Bieganie to rozmowa z własnym ciałem - słuchaj go, a dowiesz się, czego naprawdę potrzebujesz',
             'Nie ma złych dni na bieganie - są tylko dni, w których uczysz się czegoś nowego',
             'Bieganie nie tylko wzmacnia ciało, ale także oczyszcza umysł',
             'Każdy nowy kilometr to wygrana nad wątpliwościami',
             'Biegaj nie po to, by uciec od swoich problemów, ale by stawić im czoła',
             'Czasami najlepszy bieg to ten, który robisz dla siebie, a nie dla medali',
             'Biegnij ze spokojem w sercu, a pokonasz każdy dystans',
             'Bieganie uczy, że nic nie jest niemożliwe, jeśli tylko nie przestaniesz próbować',
             'Bieganie to lekcja pokory: wiesz, że nigdy nie będziesz doskonały, ale ciągle możesz stawać się lepszy',
             'Podążaj swoją drogą, biegaj w swoim rytmie i nie porównuj się do innych'     
             ]
taunts = ['Trzymaj mocno sombrero Speedy Gonzales!',
          'Prawdziwy Sonic na sterydach!',
          'Usain Bolt włączył turbo?',
          'He, he! Pędząca gazela!',
          'Biegnij, Forrest, biegnij!',
          'Przegonisz nawet Speed Racera!',
          'Coś, jak Mission Impossible?',
          ]
#
# START
#
# Generowanie płonącego napisu
st.markdown("""
    <style>
    @keyframes flameEffect {
        0% { color: orange; text-shadow: 0 0 10px red, 0 0 20px orange, 0 0 30px yellow, 0 0 40px red, 0 0 50px orange; }
        25% { color: yellow; text-shadow: 0 0 10px red, 0 0 20px orange, 0 0 30px yellow, 0 0 40px red, 0 0 50px orange; }
        50% { color: red; text-shadow: 0 0 10px orange, 0 0 20px yellow, 0 0 30px red, 0 0 40px orange, 0 0 50px yellow; }
        75% { color: orange; text-shadow: 0 0 10px red, 0 0 20px orange, 0 0 30px yellow, 0 0 40px red, 0 0 50px orange; }
        100% { color: yellow; text-shadow: 0 0 10px red, 0 0 20px orange, 0 0 30px yellow, 0 0 40px red, 0 0 50px orange; }
    }
    h1 {
        animation: flameEffect 10s infinite;
    }
    </style>
    <h1 style="text-align: center;">Wyrocznia<br>Wrocławska</h1>
""", unsafe_allow_html=True)

# Utworzenie 3 kolumn o różnej szerokości
col1, col2, col3 = st.columns([3, 5, 3])
with col1:
    st.image('candle.gif', use_container_width=True)
with col3:
    st.image('candle.gif', use_container_width=True)
with col2:
    st.markdown('<p style="color: orange;">Witaj!<br>Jestem sławną wyrocznią!<br>Dzięki mnie dowiesz się jaki osiągniesz<br>czas we Wrocławskim Półmaratonie</p>', unsafe_allow_html=True)
    # Utworzenie pola tekstwego do przyjęcia danych
    st.write(st.session_state.text)
    text = st.text_area(
        "Podaj mi tylko swoją płeć, wiek i czas na 5 km",
        height=90,
        placeholder="płeć, wiek i czas na 5 km",
        value=st.session_state.text
)    
    if text:   
        if st.button('Wysyłam!'):
            # Pobranie odpowiedzi od AI
            answer=get_data_from_text(text)
                   
            # Sprawdzenie czy w odpowiedzi są wszystkie 3 dane i reakcja na ich brak
            missing_values = [key for key, value in answer.items() if not value]
            if missing_values:
                if len(missing_values)==3:
                    st.markdown('<p style="color: orange;">Napisz coś sensownego!</p>', unsafe_allow_html=True)
                else:
                    if "age" in missing_values:
                        st.markdown('<p style="color: orange;">Pochwal się ile masz lat!</p>', unsafe_allow_html=True)
                    if "sex" in missing_values:    
                        st.markdown('<p style="color: orange;">Jeszcze nie znam Twojej płci!</p>', unsafe_allow_html=True)
                    if "time" in missing_values:    
                        st.markdown('<p style="color: orange;">Bez Twojego czasu na 5 km, wróżby nie będzie!</p>', unsafe_allow_html=True)
            else:
                # Sprawdzenie sensowności danych
                if (17 <= answer["age"] <= 99) and (14 <= answer["time"] < 90):

                    # Kilka napisów które nawzajem się zastępują
                    holder = st.empty()
                    # Losowy wybór drwiny
                    taunt = random.choice(taunts)
                    text1=f'<p style="color: orange;">{taunt}</p>'
                    holder.markdown(text1, unsafe_allow_html=True)
                    time.sleep(3)
                    # Usunięcie poprzedniego tekstu
                    holder.empty()
                    # Nowy tekst, itd.
                    text2='<p style="color: orange;">"O, przepraszam! Taki żarcik...</p>'
                    holder.markdown(text2, unsafe_allow_html=True)
                    time.sleep(3)
                    holder.empty()
                    text3='<p style="color: orange;">Jeśli się postarasz, to możesz mieć taki czas:</p>'
                    holder.markdown(text3, unsafe_allow_html=True)

                    # Utworzenie df z danymi
                    df = pd.DataFrame(answer, index=[0])

                    # Przpisanie do odpowiedniej kategorii
                    if df.loc[0, 'sex'] == 'M':
                        if df.loc[0, 'age'] < 30:
                            df['category'] = 'M20'
                        elif df.loc[0, 'age'] < 40:
                            df['category'] = 'M30'
                        elif df.loc[0, 'age'] < 50:
                            df['category'] = 'M40'
                        elif df.loc[0, 'age'] < 60:
                            df['category'] = 'M50'
                        elif df.loc[0, 'age'] < 70:
                            df['category'] = 'M60'
                        elif df.loc[0, 'age'] < 80:
                            df['category'] = 'M70'
                        else:
                            df['category'] = 'M80'
                    elif df.loc[0, 'sex'] == 'F':
                        if df.loc[0, 'age'] < 30:
                            df['category'] = 'K20'
                        elif df.loc[0, 'age'] < 40:
                            df['category'] = 'K30'
                        elif df.loc[0, 'age'] < 50:
                            df['category'] = 'K40'
                        elif df.loc[0, 'age'] < 60:
                            df['category'] = 'K50'
                        elif df.loc[0, 'age'] < 70:
                            df['category'] = 'K60'
                        elif df.loc[0, 'age'] >= 70:
                            df['category'] = 'K70'    

                    # Dostosowanie df do modelu   
                    df['sex'] = df['sex'].replace('F', 'K')
                    df = df.drop('age', axis=1)
                    df.columns = ['Płeć', '5 km Czas', 'Kategoria wiekowa']
                    df['5 km Czas'] = df['5 km Czas']* 60
                    df['5 km Czas'] = df['5 km Czas'].astype('float64')
                    
                    # Wykonanie predykcji i odczyt wyniku 
                    prediction = predict_model(model, data=df)
                    exp_time = prediction.loc[0,'prediction_label']
                    hour = str(int(exp_time // 3600))
                    min = str(int(exp_time % 3600) // 60)
                    sec = str(int(exp_time % 60))

                    # Utworzenie 3 kolumn do wyświetlenia wyniku
                    sc1, sc2, sc3 = st.columns(3)
                    time.sleep(2)
                    with sc1:
                        st.markdown(f'<p style="color: red; font-size: 19px; font-weight: bold">godzin: {hour}</p>', unsafe_allow_html=True)
                    time.sleep(2)
                    with sc2:
                        st.markdown(f'<p style="color: red; font-size: 19px; font-weight: bold">minut: {min}</p>', unsafe_allow_html=True)
                    time.sleep(2)
                    with sc3:
                        st.markdown(f'<p style="color: red; font-size: 19px; font-weight: bold">sekund: {sec}</p>', unsafe_allow_html=True)
                    time.sleep(2)

                    # Losowy wybór mądrości i wyświetlenie jej
                    sentence = random.choice(sentences)
                    with sc2:
                        st.markdown('<p style="color: orange; text-align: center; font-size: 18px; font-weight: bold">Pamiętaj!</p>', unsafe_allow_html=True)
                    time.sleep(2)
                    st.markdown(f'<p style="color: orange; text-align: center; font-size: 18px">{sentence}</p>', unsafe_allow_html=True)
                    time.sleep(5)
                    
                    # Reset aplikacji
                    if st.button("RESET"):
                        st.session_state.text = ""
                        st.experimental_rerun()
                       
                else:
                    if (answer["time"] >= 90) or (answer["age"] > 99):
                        st.markdown('<p style="color: orange;">Obawiam się, że dojedziesz do mety pojazdem "koniec biegu"</p>', unsafe_allow_html=True)
                    if answer["time"] < 14:
                        st.markdown('<p style="color: orange;">REKORD ŚWIATA! Ten czas, to chyba pomyłka?</p>', unsafe_allow_html=True)
                    if answer["age"] < 17:
                        st.markdown('<p style="color: orange;">W biegu mogą uczestniczyć tylko osoby dorosłe!</p>', unsafe_allow_html=True)

        else:
            st.markdown('<p style="color: orange;">Czy chcesz mi wysłać te informacje?</p>', unsafe_allow_html=True)
            
    else:
        time.sleep(5)
        st.markdown('<p style="color: orange;">Czekam i czekam, myślę i myślę...</p>', unsafe_allow_html=True)  
 




  

        
        