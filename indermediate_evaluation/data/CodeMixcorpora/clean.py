import re

file=open('train_csnliProc.txt','r')
test = file.read()

test=re.sub(r'(<unk>|<उनक>)', '', test)   

file.close()
file2=open("train_clean.txt",'w')
file2.write(test)
file2.close()


