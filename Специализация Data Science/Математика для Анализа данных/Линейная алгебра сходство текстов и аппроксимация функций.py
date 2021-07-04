#!/usr/bin/env python
# coding: utf-8

# # Задача 1: Сравнение предложений

# Дан набор предложений, скопированных с Википедии. Каждое из них имеет "кошачью тему" в одном из трех смыслов:
# 
# * кошки (животные) 
# * UNIX-утилита cat для вывода содержимого файлов   
# * версии операционной системы OS X, названные в честь семейства кошачьих  
# 
# *Задача — найти два предложения, которые ближе всего по смыслу к расположенному в самой первой строке. В качестве меры близости по смыслу мы будем использовать косинусное расстояние.*
# 
# 

# Выполните следующие шаги:
# 
# 1. Скачаем файл.  
# 2. Каждая строка в файле соответствует одному предложению. Считаем их и приведём каждую к нижнему регистру.   
# 3. Произведём токенизацию, то есть разбиение текстов на слова. Удалим пустые слова после разделения.  
# 4. Сопоставим список всех слов, встречающихся в предложениях.     
# 5. Создадим матрицу размера n * d, где n — число предложений. Заполним ее: элемент с индексом (i, j) в этой матрице должен быть равен количеству вхождений j-го слова в i-е предложение. У нас должна получиться матрица размера 22 * 254.  
# 6. Найдите косинусное расстояние от предложения в самой первой строке до всех остальных. Найдём две строки, ближайших к этому расстоянию?  
# 
#   

# ### Скачаем файл с предложениями.

# In[1]:


sentences = open(r'sentences.txt', 'r+').readlines()


# ### Каждая строка в файле соответствует одному предложению. Считаем их и приведём каждую к нижнему регистру.

# In[2]:


sentences = [x.lower() for x in sentences]
sentences = [x.split('\n')[0] for x in sentences]


# In[3]:


sentences


# ### Произведём токенизацию.

# In[4]:


import re


# In[5]:


sentences = [re.split('[^a-z]',sentences[x]) for x in range(len(sentences))]


# In[6]:


for i in range(len(sentences)):
    while '' in sentences[i]:
        sentences[i].remove('') 


# ### Составьте список всех слов, встречающихся в предложениях.

# In[7]:


words = []
for row in range(len(sentences)):
    for word in sentences[row]:
        if word not in words:
            words.append(word)


# In[8]:


dct = {}
for i in range(len(words)):
    dct[i] = words[i]


# In[9]:


dct


# ### Создадим матрицу.

# In[5]:


import numpy as np


# In[13]:


matrix = np.zeros((22,254))


# In[14]:


matrix


# In[15]:


for i in range(len(sentences)):
    for j in dct.keys(): 
        if dct[j] in sentences[i]:
            matrix[i, j] += 1 


# In[17]:


matrix[10]


# ### Найдите косинусное расстояние от предложения в самой первой строке  до всех остальных.

# In[18]:


from scipy.spatial.distance import cosine


# In[19]:


cosines = {}
for i in range(1, len(matrix)):
    cos = cosine(matrix[0], matrix[i])
    print(f'Косинусное расстояние между первой строкой и строкой {i}: {cosine(matrix[0], matrix[i])}')
    cosines[i] = cos


# #### Минимумы

# In[20]:


sorted(cosines.values())[0:2]


# In[21]:


for key in cosines.keys():
    if (cosines[key] == 0.7547442642060137) or (cosines[key] == 0.8055388829343507):
        print(key)


# In[22]:


print(open(r'sentences.txt', 'r+').readlines()[0])
print(open(r'sentences.txt', 'r+').readlines()[4])
print(open(r'sentences.txt', 'r+').readlines()[6])


# In[35]:


ans = [str(4), str(6)]


# In[36]:


with open('answer_1', 'w') as output_file:
    output_file.write(' '.join(ans))


# # Задача №2

# Рассмотрим сложную математическую функцию на отрезке [1, 15]:
# 
# f(x) = sin(x / 5) * exp(x / 10) + 5 * exp(-x / 2)
# 
# 

# In[58]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import sin, exp


# In[59]:


sns.set(
    font_scale=1,
    style='whitegrid',
    rc={'figure.figsize':(10,8)}
)


# In[82]:


f = lambda x: sin(x/5) * exp(x/10) + 5 * exp(-x/2)


# In[83]:


a = np.linspace(1, 15, 30)


# In[84]:


plt.plot(a, f(a));


# Как известно, многочлен степени n (то есть w_0 + w_1 x + w_2 x^2 + ... + w_n x^n) однозначно определяется любыми n + 1 различными точками, через которые он проходит. Это значит, что его коэффициенты w_0, ... w_n можно определить из следующей системы линейных уравнений:

# ![image.png](attachment:image.png)

# где через x_1, ..., x_n, x_{n+1} обозначены точки, через которые проходит многочлен, а через f(x_1), ..., f(x_n), f(x_{n+1}) — значения, которые он должен принимать в этих точках.

# ### 1. Сформируем систему линейных уравнений для многочлена первой степени, который должен совпадать с функцией f в точках 1 и 15.  
# 

# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np;
import math;
import matplotlib.pyplot as plt;



def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)



# approximate at the given points (feel free to experiment: change/add/remove)
points = np.array([1, 4, 10, 15])
n = points.size



# fill A-matrix, each row is 1 or xi^0, xi^1, xi^2, xi^3 .. xi^n
A = np.zeros((n, n))
for index in range(0, n):
    A[index] = np.power(np.full(n, points[index]), np.arange(0, n, 1))



# fill b-matrix, i.e. function value at the given points
b = f(points)



# solve to get approximation polynomial coefficents
solve = np.linalg.solve(A,b)



# define the polynome approximation of the function
def polinom(x): 
    # Yi = solve * Xi where Xi = x^i
    tiles = np.tile(x, (n, 1))
    tiles[0] = np.ones(x.size)
    for index in range(1, n):
        tiles[index] = tiles[index]**index
    return solve.dot(tiles)



# plot the graphs of original function and its approximation
x = np.linspace(1, 15, 100)
plt.plot(x, f(x))
plt.plot(x, polinom(x))



# print out the coefficients of polynome approximating our function
print(solve)


# In[96]:


A, b


# In[ ]:




