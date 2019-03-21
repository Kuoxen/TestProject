import random
class vector(object):

    def __init__(self,arr):
      self.arr = arr
      self.len=len(arr)

    def copyFrom(self,arr,low,high):
        self.arr=arr[low:high+1]

    def permute(self):
        i=self.len-1
        while(i>=0):
            rand=random.randint(0,i-1) 
            self.arr[i],self.arr[rand]=self.arr[rand],self.arr[i]
            i-=1
        
    def unsort(self,low,high):
        i=high-low 
        while(i>=0):
            rand=random.randint(0,i-1) 
            self.arr[i+low],self.arr[rand+low]=self.arr[rand+low],self.arr[i+low]
            i-=1

    def find(self,v,low,high):
        pass