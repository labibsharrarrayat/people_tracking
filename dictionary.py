import time
from itertools import cycle

Dictionary1 = {1: 'Geeks', 2: 'Britain', 3: 'Geeks'}

print("Original Dictionary items:")

items = Dictionary1.items()

# Printing all the items of the Dictionary
print(items)

# Delete an item from dictionary
#del [Dictionary1['C']]
print('Updated Dictionary:')
print(Dictionary1.keys())


vals = []
count = 0

def go_to(d, c):

    print(c)

    for i in range(len(d)):

        if(c <= 10):
            i = 0

        print(d[i])




while True:

    for i in Dictionary1.keys():

        for v in Dictionary1.keys():
            vals.append(v)

        go_to(vals,count)
        count = count + 1

        vals = []
        time.sleep(1)
