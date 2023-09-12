def make99():
    for i in range(1,10,1):
        print('')
        for j in range(1,10):
            ans=i*j
            if((i*j)%2==1):
                print(ans,'*',end='\t')
            else:
                print(ans,end='\t') 