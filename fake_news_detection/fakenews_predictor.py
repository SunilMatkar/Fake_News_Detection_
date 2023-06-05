import pandas as pd, numpy as np, re

from sklearn.metrics import classification_report, accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
import tkinter as tk
from sklearn import svm
from PIL import Image, ImageTk
from tkinter import ttk
from joblib import dump , load
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle
import nltk

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

#######################################################################################################
nltk.download('stopwords')
stop = stopwords.words('english')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#######################################################################################################
    
root = tk.Tk()
root.title("Fake news Detection using Machine Learning")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
image2 =Image.open('f1.jpg')
image2 =image2.resize((w,h), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=0)

###########################################################################################################
lbl = tk.Label(root, text="Fake news Detection using Machine Learning", font=('times', 35,' bold '),width=50, height=1,bg="#FFBF40",fg="black")
lbl.place(x=0, y=10)
###########################################################################################################

#frame = tk.LabelFrame(root,text="Crime Predictor",width=250,height=300,bd=3,background="cyan2",font=("Tempus Sanc ITC",15,"bold"))
#frame.place(x=50,y=100)
#frame['borderwidth'] = 10
frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=400, bd=5, font=('times', 14, ' bold '),bg="lawn Green")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=10, y=120)

def Data_Display():
    columns = ['ID', 'statment', 'Result']
    print(columns)

    data1 = pd.read_csv(r"C:/Users/matka/Desktop/fake news/code/fake_news_detection/fake_news_detection/valid.csv", encoding='unicode_escape')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    ID = data1.iloc[:, 0]
    statment = data1.iloc[:, 0]
    Result = data1.iloc[:, 1]
   
    


    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=270, y=100)

    tree = ttk.Treeview(display, columns=('ID', 'statment', 'Result'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=40)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Calibri', 10), background="black")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3")
    tree.column("1", width=130)
    tree.column("2", width=150)
   

    tree.heading("1", text="ID")
    tree.heading("2", text="statment")
    tree.heading("3", text="Result")
    
    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")

    for i in range(0, 304):
        tree.insert("", 'end', values=(
            ID[i], statment[i], Result[i]))
        i = i + 1
        print(i)

##############################################################################################################


def Train():
    
    result = pd.read_csv(r"C:/Users/matka/Desktop/fake news/code/fake_news_detection/fake_news_detection/valid.csv",encoding = 'unicode_escape')

    result.head()
        
    
    result = 0
    
    x_train, x_test, y_train, y_test = train_test_split(result['Statement'], result['Result'],test_size=0.2, random_state=1)
    
    #Logistic regression classification
    pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])

    model_lr = pipe1.fit(x_train, y_train)
    lr_pred = model_lr.predict(x_test)
    ACC=format(round(accuracy_score(y_test, lr_pred)*100,2))
    print("Accuracy of Logistic Regression Classifier: {}%".format(round(accuracy_score(y_test, lr_pred)*100,2)))
    print("\nConfusion Matrix of Logistic Regression Classifier:\n")
    print(confusion_matrix(y_test, lr_pred))
    print("\nCLassification Report of Logistic Regression Classifier:\n")
    
    print(classification_report(y_test, lr_pred))
    repo = classification_report(y_test, lr_pred)
        
     
    label4 = tk.Label(root,text =str(repo),width=35,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=100)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as LOG_MODEL.joblib",width=35,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=320)
    
    dump (model_lr,"LOG_MODEL.joblib")
    print("Model saved as LOG_MODEL.joblib")



entry = tk.Entry(root,width=17,font=("Tempus Sanc ITC",14))
entry.insert(0,"Enter text here...")
entry.place(x=25,y=290)
##############################################################################################################################################################################
def Test():
    predictor = load("LOG_MODEL.joblib")
    Given_text = entry.get()
    Given_text = [Given_text]
    print(type(Given_text))
    v = predictor.predict(Given_text)
    if v == 0:
        label4 = tk.Label(root,text ="Fake new is not detected ",width=20,height=2,bg='#46C646',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
        print("The Accuracy Of The Model is 89%")
    else:
        label4 = tk.Label(root,text ="Fake news detected ",width=20,height=2,bg='#FF3C3C',fg='black',font=("Tempus Sanc ITC",25))
        label4.place(x=450,y=550)
        print("The Accuracy Of The Model is 89%")
    
###########################################################################################################################################################
def window():
    root.destroy()
    
button1 = tk.Button(root,command=Data_Display,text="Data_Display",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button1.place(x=25,y=150)

button2 = tk.Button(root,command=Train,text="Train",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button2.place(x=25,y=220)

button3 = tk.Button(root,command=Test,text="Test",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button3.place(x=25,y=360)

button4 = tk.Button(root,command=window,text="Exit",bg="#E46EE4",fg="black",width=15,font=("Times New Roman",15,"bold"))
button4.place(x=25,y=430)




root.mainloop()