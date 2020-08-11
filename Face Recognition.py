#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
import tkinter as tk
from tkinter import *


# In[2]:


window = Tk()
window.geometry('600x600')
window.configure(bg='white')
window.title('Face detection')
lbl1 = Label(window , text = "Face detection",fg = "white" ,bg ="red", width =30 ,height =3,font =('times', 25, ' bold '))
lbl1.place(x=0 , y=0)

lbl2 = Label(window , text = "Enter Image Name" ,bg='white', fg = "red" , font = ('times', 15 ,'bold')).place(x=100 , y=200)

lbl3 = Label(window, text = "Live Detection", fg = "red" ,bg='white', font = ('times', 15 ,'bold')).place(x=100 ,y=400)
entry1 = tk.Entry(window , bg ="red" ,fg ="white" ,width =30)
entry1.place(x=90,y=240)


# In[3]:


def onClick1():
    image_path = entry1.get()
    if image_path == "":
        lbl5 = Label(window, text ="Enter Name of Image", fg="red").place(x=200 ,y=280)
    cascpath= "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascpath)
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(10, 10))
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w ,y+h), (0, 255, 0), 2)
            
    cv2.imshow("images found" , image)
    cv2.waitKey(0)
    
btn1 = Button(window, text = "Click", command = onClick1 ,fg ='white', bg = "red",font = ('times', 15 ,'bold'), width =10).place(x =350 , y=  230)


# In[4]:


def live():
    cam = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    while(True):
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
            
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(10, 10))
            
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y) ,(x+w , y+h), (0,255,0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Found',(x,y), font, 1, (255,0,0))
                
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
                
            
    cam.release()
    cv2.destroyAllWindows()

btn2 = Button(window, text = "Live", command = live ,fg = "white", bg = "red",font = ('times', 15 ,'bold'), width =10).place(x = 350 , y = 400)


# In[5]:


window.mainloop()

