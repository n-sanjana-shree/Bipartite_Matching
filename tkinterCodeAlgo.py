import pandas
import tkinter as tk
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

w = []
gl=0

data = pd.read_csv("D:/Algorithm project sem1/flipkart_com-ecommerce_sample.csv")
products = list(data.iloc[:, 0])
users = list(data.iloc[:, 1])

w = list(data.iloc[:, 2])
E = zip(products,users, w)
attributes = pd.DataFrame([{"source": products, "target": users, "weights": weight} for users, products, weight in E])

G = nx.Graph()

G = nx.from_pandas_edgelist(attributes, 'source', 'target', edge_attr='weights')


pos = nx.spring_layout(G,k=1000)
pos.update( (n, (1, i)) for i, n in enumerate(products) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(users) ) # put nodes from Y at x=2
labels = nx.get_edge_attributes(G, 'weights')
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='orange', arrowsize=30)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)




plt.show()


from itertools import islice

ar = [0] * 5
l = []

for i in products:
    for j in users:
        ans = G.get_edge_data(i, j, default=0)
        if (ans == 0):
            ans1 = 0

        else:
            ans1 = G.get_edge_data(i, j, default=0)['weights']

        l.append(ans1)

length_to_split = [26]*26
Inputt = iter(l)
Output = [list(islice(Inputt, elem))
          for elem in length_to_split]


from typing import List, Tuple
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_sum_assignment_brute_force(
        cost_matrix: np.ndarray,
        maximize: bool = False) -> Tuple[List[int], List[int]]:

    h = cost_matrix.shape[0]
    w = cost_matrix.shape[1]

    if maximize is True:
        cost_matrix = -cost_matrix

    minimum_cost = float("inf")

    if h >= w:
        for i_idices in itertools.permutations(list(range(h)), min(h, w)):
            row_ind = i_idices
            col_ind = list(range(w))
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < minimum_cost:
                minimum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind
    if h < w:
        for j_idices in itertools.permutations(list(range(w)), min(h, w)):
            row_ind = list(range(h))
            col_ind = j_idices
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < minimum_cost:
                minimum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind

    return optimal_row_ind, optimal_col_ind

if __name__ == "__main__":
    r=3
    cost_matrix = np.array(Output)

    row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix,
                                             maximize=True)

    minimum_cost = cost_matrix[row_ind, col_ind].sum()

    print(
        f"The optimal assignment from Hungarian algorithm is: {list(zip(row_ind, col_ind))}."
    )

    print(f"The maximum cost from Hungarian algorithm is: {minimum_cost}.")

    ll=[]
    for i in range(len(row_ind)):
        ll.append(row_ind[i])
        ll.append(col_ind[i])

    length_to_split = [2] * 26
    Inputt = iter(ll)
    Output = [list(islice(Inputt, elem))
                  for elem in length_to_split]

    cost=[]
    lst1=[]
    for i in range(len(row_ind)):
            a=row_ind[i]
            b=col_ind[i]
            lst1.append(cost_matrix[a][b])
    lll=[]
    for i in range(len(Output)):
        for j in range(2):
            if(j==0):
                lll.append(products[Output[i][j]])
            elif(j==1):
                lll.append(users[Output[i][j]])
    l1=[]
    fl1=[]

    for i in range(len(cost_matrix)):
        l1=[]
        for j in range(len(cost_matrix[0])):
            if(cost_matrix[i][j]!=0):
                usr=j
                l1.append(users[usr])
        fl1.append(l1)
    length_to_split = [2] * 26
    Inputt = iter(lll)
    Output = [list(islice(Inputt, elem))
                  for elem in length_to_split]
    number_of_sale=len(lst1)
    for i in range(len(Output)):
            Output[i].append(lst1[i])

    cost_list=[]
    for i in range(len(Output)):
        cost_list.append(Output[i][2])
    indexes_profit_costs_prod=[]
    for i in range(len(cost_list)):
        if(cost_list[i]==max(cost_list)):
            indexes_profit_costs_prod=i

    csvFile1 = pandas.read_csv('C:/Users/Sanjana/Desktop/ALGO LAB/flipkart_com-ecommerce_sample.csv')
    l9 = csvFile1.values.tolist()
    l91=[]

    l91.append(l9[indexes_profit_costs_prod][0])
    l91.append(l9[indexes_profit_costs_prod][3])
    l91.append(l9[indexes_profit_costs_prod][2])
    res=[]
    n=3
    res=[l91[i:i + n] for i in range(0, len(l91), n)]


    from tkinter import *
    from tkinter import ttk
    csvFile = pandas.read_csv('C:/Users/Sanjana/Desktop/ALGO LAB/flipkart_com-ecommerce_sample.csv')
    l=csvFile.values.tolist()


    win=Tk()
    win.maxsize(1900,1200)
    win.minsize(1900,1200)
    win.title("Bipartite Matching")

    bg = tk.PhotoImage(file ="C:/Users/Sanjana/Desktop/ALGO LAB/background2.png")
    label1 =tk.Label(image = bg)
    label1.place(x = 0,y = 0)


    head=Label(win,text="WELCOME BUYERS",background='#FFFFFA',height=2,width=30,font=('Franklin Gothic Medium',20),anchor='center')
    head.place(relx=0.35,rely=0.05)

    productTitle=Label(win,text='PRODUCTS',background='DarkSlateGray1',height=1,width=25,font=('Georgia',16))
    productTitle.place(relx=0.05,rely=0.15)

    usersTitle=Label(win,text='USERS',background='DarkSlateGray1',height=1,width=18,font=('Georgia',16))
    usersTitle.place(relx=0.35,rely=0.15)

    priceTitle=Label(win,text='PRICE',background='DarkSlateGray1',height=1,width=18,font=('Georgia',16))
    priceTitle.place(relx=0.55,rely=0.15)

    brandTitle=Label(win,text='BRAND',background='DarkSlateGray1',height=1,width=18,font=('Georgia',16))
    brandTitle.place(relx=0.75,rely=0.15)

    productBody=Listbox(win,height=12,width=25,bg='#7F7FFF',font=('Segoe Print',14))
    productBody.place(relx=0.05,rely=0.20)
    for i in range(len(l)):
        productBody.insert(i+1,l[i][0])



    usersBody=Listbox(win,height=12,width=17,bg='#7F7FFF',font=('Segoe Print',14))
    usersBody.place(relx=0.35,rely=0.20)
    for i in range(len(l)):
        usersBody.insert(i+1,l[i][1])



    priceBody=Listbox(win,height=12,width=17,bg='#7F7FFF',font=('Segoe Print',14))
    priceBody.place(relx=0.55,rely=0.20)
    for i in range(len(l)):
        priceBody.insert(i+1,l[i][2])


    brandBody=Listbox(win,height=12,width=17,bg='#7F7FFF',font=('Segoe Print',14))
    brandBody.place(relx=0.75,rely=0.20)
    for i in range(len(l)):
        brandBody.insert(i+1,l[i][3])

    def newwindow():

        new = Toplevel()
        new.maxsize(1900,1200)
        new.minsize(1900,1200)
        main_frame=Frame(new)
        main_frame.pack(fill=BOTH,expand=1)
        my_canvas=Canvas(main_frame)
        my_canvas.pack(side=LEFT,fill=BOTH,expand=1)
        my_scrollbar=ttk.Scrollbar(main_frame,orient=VERTICAL,command=my_canvas.yview)
        my_scrollbar.pack(side=RIGHT,fill=Y)
        bg1 = tk.PhotoImage(file="C:/Users/Sanjana/Desktop/ALGO LAB/background2.png")
        my_canvas.configure(yscrollcommand=my_scrollbar.set)
        my_canvas.bind('<Configure>',lambda e:my_canvas.configure(scrollregion=my_canvas.bbox("all")))
        second_frame=Frame(my_canvas,background='#141499')
        my_canvas.create_window((0,0),window=second_frame)
        head= Label(second_frame, text='FIND YOUR BEST CUSTOMER', background='white', height=1, width=40,
                         font=('Calibri', 25)).grid(row=3, column=2, pady=100)

        def open_popup1():
            top = Toplevel(win,background='#87CEEB')
            top.geometry("1700x200")
            top.title("Summary window")
            Label(top,text=f'The {l91[0]} product is the costliest with price Rs {l91[2]}',
                      font=('Georgia',20),background='#87CEEB').grid(row=3, column=5,padx=50,pady=60)


        def open_popup2():
            top = Toplevel(win,background='#87CEEB')
            top.geometry("1000x200")
            top.title("Summary window")

            Label(top, text=f'\n{l91[1]} is the supplier with maximum profit',
                      font=('Georgia',20),background='#87CEEB').grid(row=3, column=5,padx=50,pady=40)

        def open_popup4():
            second_frame.destroy()
            new.destroy()

        def open_popup5():
            win.destroy()

        def open_popup6():
            top = Toplevel(win, background='#87CEEB')
            top.geometry("700x200")
            top.title("Summary window")
            Label(top, text=f'\n {number_of_sale} is the total number of sales ',
                  font=('Georgia', 20), background='#87CEEB').grid(row=3, column=5, padx=50, pady=40)

        def open_popup7():
            top = Toplevel(win, background='#87CEEB')
            top.geometry("1100x200")
            top.title("Summary window")
            Label(top, text=f'\n Rs {minimum_cost} is the total cost of sales including all customers ',
                  font=('Georgia', 20), background='#87CEEB').grid(row=3, column=5, padx=50, pady=40)


        headButton = Button(second_frame, text='Costliest Product', background='#d4ebf2',activebackground='green', height=1, width=20,
                     font=('Arial', 16),command=open_popup1).grid(row=4, column=3)
        headButton = Button(second_frame, text='Best Supplier', background='#d4ebf2', activebackground='green', height=1,
                            width=20,
                            font=('Arial', 16), command=open_popup2).grid(row=5, column=3)

        headButton = Button(second_frame, text='Total Number of Sales', background='#d4ebf2', activebackground='green',
                            height=1,
                            width=20,
                            font=('Arial', 16), command=open_popup6).grid(row=6, column=3)
        eadButton = Button(second_frame, text='Total Sales Price', background='#d4ebf2', activebackground='green',
                           height=1,
                           width=20,
                           font=('Arial', 16), command=open_popup7).grid(row=7, column=3)
        eadButton = Button(second_frame, text='Back', background='#d4ebf2', activebackground='green',
                           height=1,
                           width=20,
                           font=('Arial', 16), command=open_popup4).grid(row=8, column=3)
        headButton = Button(second_frame, text='Exit', background='#d4ebf2', activebackground='green',
                            height=1,
                            width=20,
                            font=('Arial', 16), command=open_popup5).grid(row=9, column=3)
        r = 3
        ind=0
        for k in range(0,25):
            var = Output[k][0]

            r=r+1
            ind=ind+1
            request3 = Label(second_frame, text=f'{ind}) {var} is requested by', background='#141499', height=1, width=80,
                         font=('Segoe Print', 16),fg='white').grid(row=r, column=2, pady=20,padx=20)
            temp_var = ""
            for i in range(len(fl1[k])):
                temp_var += fl1[k][i]
                temp_var += " "

            r=r+1
            userTitle3 = Label(second_frame, text=f'{temp_var}', background='#141499', height=1, width=80,activebackground='green',
                           font=('Segoe Print', 16),fg='white').grid(row=r, column=2, pady=20)

            def show(r):
                top = Toplevel(win,background='#7DF9FF')
                top.geometry("2050x200")
                top.title("Result Window")

                Label(top, text=f'{Output[r][0]} can be sold to {Output[r][1]} for Rs {Output[r][2]} to gain maximum profit ', font=('Georgia',20),background='#7DF9FF').place(x=60, y=80)

            r=r+1
            from functools import partial
            submit3 = Button(second_frame, text='Best suited customer', background='#d4ebf2',activebackground='green', height=1, width=20, font=('Arial', 16),
                        command=partial(show, k)).grid(row=r, column=2, pady=20)
    def open_popup5():

        win.destroy()
    submit=Button(win,text='Next',background='#d4ebf2',height=1,width=10,font=('Arial',16),activebackground='green',command=newwindow)
    submit.place(relx=0.41,rely=0.73)
    submit = Button(win, text='Exit', background='#d4ebf2', height=1, width=10, font=('Arial', 16),
                    activebackground='green', command=open_popup5)
    submit.place(relx=0.54, rely=0.73)
    win.mainloop()
































