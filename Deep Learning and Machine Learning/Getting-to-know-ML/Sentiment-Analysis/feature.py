x = []
y = []
for value in lines:
    temp = value.split('\t')
    x.append(temp[0])
    temp[1].replace('\n','')
    y.append(int(temp[1]))
