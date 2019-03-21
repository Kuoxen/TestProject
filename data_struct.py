##### sort
def bubble_sort(arr):
    round=0
    while(round<=len(arr)-1):
        flag=0
        for i in range(1,len(arr)-round):
            if arr[i]<arr[i-1]:
                arr[i-1],arr[i]=arr[i],arr[i-1]
                flag=1
        if flag==0:
            break
        round+=1
####### count one 
def count_ones(i):
    tmp=i
    count=0
    while(tmp>0):
        count+=tmp&1
        tmp=tmp>>1
    return count

###### reverse
def reverse_by_recursion(arr,n):
    if n==1:
        return [arr[n-1]]
    else:
        return [arr[n-1]]+reverse_by_recursion(arr,n-1)

def reverse_by_recursion(arr,low,high):
    if low<high:
        arr[low],arr[high]=arr[high],arr[low]
        reverse_by_recursion(arr,low+1,high-1)

##### power2
def power_by_recursion(n):
    if n==0:
        return 1
    else:
        return 2*power_by_recursion(n-1)

def power_by_recursion(n):
    if n==0:
        return 1
    elif n%2==1:
        return power_by_recursion(n//2)*power_by_recursion(n//2)*2
    else:
        return power_by_recursion(n//2)*power_by_recursion(n//2)


#### fibonacci

def fibonacci(n,prev):
    if n==0:
        prev[0]=1
        return 0
    else:
        prevprev=[0]
        prev[0]=fibonacci(n-1,prevprev)
        return prev[0]+prevprev[0]


prev=[0]
fibonacci(1,prev)


def fibonacci(n):
    a=0
    b=1
    while(n>0):
        b+=a
        a=b-a
        n-=1
    return b




arr=[3,2,1]
bubble_sort(arr)
reverse_by_recursion(arr,3)
reverse_by_recursion(arr,0,2)