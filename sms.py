import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download("punkt")
nltk.download('stopwords')

ps = PorterStemmer()


def preprocess_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))


  return " ".join(y)


tfdf = pickle.load(open('tfvectorizer.pkl','rb'))
model = pickle.load(open('modeltf.pkl','rb'))


activities = ["Home", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)

st.sidebar.markdown(
        """ Developed by Sampanna  
            Email : mishrasampanna1998@gmail.com  
            [LinkedIn] (http://www.linkedin.com/in/sampanna-mishra-0996ab18b/)
            """)

if choice == "Home":
        html_temp_home1 = """<div style="background-color:#4d1547;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Spam Message Classifier application using NLP, Naive Bayes algorithm and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)

elif choice == "About":

    st.subheader("About this app")
    html_temp_about1= """<div style="background-color:#e0cce3;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Spam  Test Detector application using NLP, Naive Bayes algorithm and Streamlit.</h4>
                                    </div>
                                    </br>"""
    st.markdown(html_temp_about1, unsafe_allow_html=True)

    html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Sampanna Mishra using Streamlit Framework, NLP library and Machine Learning classifier algorithm for demonstration purpose.If you have any suggestion or wnat to comment just write a mail at mishrasampanna1998@gmail.com. </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

    st.markdown(html_temp4, unsafe_allow_html=True)

else:
  pass

st.title('Spam Message Classifier')
input_sms = st.text_input('Enter the message')

if st.button('Predict'):

# preprocess
    transformed_text = preprocess_text(input_sms)
# vectorize
    vector_input = tfdf.transform([transformed_text])
# predict
    result = model.predict(vector_input)[0]

# display
    if result == 1:

        st.header('Spam')
    else:
        st.header('Not Spam')

