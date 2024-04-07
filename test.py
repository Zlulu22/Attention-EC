
loss_list = [] 
def compute(i):
    loss = i
    
    loss_list.append(loss)
    print(loss_list)

for i in range(10):
    compute(i)
