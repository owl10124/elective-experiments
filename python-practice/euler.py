t={}

m=100000
b=0

def ord(x):
    y=0
    while (x):
        x//=10
        y+=1
    return y

def dig(x,i): /*from 0 to*/
    return x/(10**i)%10

for i in range(2,m):
    if (1<<i)&b: continue
    x = ord(i)-1
    for j in range(1<<x):
        y=-1
        s=True
        for k in range(x):
            if (j&(1<<k)):
                if(y==-1) y=dig(i,k)
                elif (y!=dig(i,k)) s=False
        if not s: continue
        if not j in t:

    for j in range(i*i,m,i):
        if not (1<<j)&b: b|=1<<j
