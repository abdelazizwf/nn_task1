import tkinter as tk
from tkinter import *

root = Tk()
root.title('Penguin classification')
root.geometry("900x500")


def upd1():
    c=0
    if(c4_v.get()==1):c=c+1
    if(c5_v.get()==1):c=c+1    
    if(c6_v.get()==1):c=c+1
    if(c7_v.get()==1):c=c+1
    if(c8_v.get()==1):c=c+1
    if(c>=2):
        if(c4_v.get()!=1):c4.config(state='disabled')
        if(c5_v.get()!=1):c5.config(state='disabled')
        if(c6_v.get()!=1):c6.config(state='disabled')
        if(c7_v.get()!=1):c7.config(state='disabled')
        if(c8_v.get()!=1):c8.config(state='disabled')
    else:
        c4.config(state='normal')
        c5.config(state='normal')
        c6.config(state='normal')
        c7.config(state='normal')
        c8.config(state='normal')
        
myLabel= Label(root, text = "please select the two features")
myLabel.grid(column = 0, row = 0, sticky=W)

c4_v=tk.IntVar(root)
c4 = tk.Checkbutton(root, text='bill length', command=upd1, variable = c4_v)
c4.grid(column = 1, row = 1)
c5_v=tk.IntVar(root)
c5 = tk.Checkbutton(root, text='bill depth', command=upd1, variable = c5_v)
c5.grid(column = 2, row = 1)
c6_v=tk.IntVar(root)
c6 = tk.Checkbutton(root, text='flipper length',command=upd1, variable = c6_v)
c6.grid(column = 3, row = 1)
c7_v=tk.IntVar(root)
c7 = tk.Checkbutton(root, text='gender', command=upd1, variable = c7_v)
c7.grid(column = 4, row = 1)
c8_v=tk.IntVar(root)
c8 = tk.Checkbutton(root, text='body mass', command=upd1, variable = c8_v)
c8.grid(column = 5, row = 1)

def upd():
    i=0
    if(c1_v.get()==1):i=i+1
    if(c2_v.get()==1):i=i+1    
    if(c3_v.get()==1):i=i+1
    if(i>=2):
        if(c1_v.get()!=1):c1.config(state='disabled')
        if(c2_v.get()!=1):c2.config(state='disabled')
        if(c3_v.get()!=1):c3.config(state='disabled')
    else:
        c1.config(state='normal')
        c2.config(state='normal')
        c3.config(state='normal')
        
myLabelC= Label(root, text = "please select the two classes")
myLabelC.grid(column = 0, row = 2, sticky=W)

c1_v=tk.IntVar(root)
c1 = tk.Checkbutton(root, text='Adelie', command=upd, variable = c1_v)
c1.grid(column = 1, row = 3)
c2_v=tk.IntVar(root)
c2 = tk.Checkbutton(root, text='Gentoo', command=upd, variable = c2_v)
c2.grid(column = 2, row = 3)
c3_v=tk.IntVar(root)
c3 = tk.Checkbutton(root, text='Chinstrap', command=upd, variable = c3_v)
c3.grid(column = 3, row = 3)

myLabelT1 = Label(root, text = "Enter learning rate: ")
myLabelT1.grid(column = 0, row = 5, sticky=W)
T1 = Text(root, height = 1, width = 10)
T1.grid(column = 1, row = 5)

myLabelT2 = Label(root, text = "Enter number of epochs: ")
myLabelT2.grid(column = 0, row = 6, sticky=W)
T2 = Text(root, height = 1, width = 10)
T2.grid(column = 1, row = 6)

c9 = tk.Checkbutton(root, text='Biased')
c9.grid(column = 3, row = 6)

root.mainloop()