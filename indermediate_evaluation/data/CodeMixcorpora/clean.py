import re

file=open('test_csnli.txt','r')
test = file.read()

test=re.sub(r'(<unk>|<उनक>)', '', test)   

file.close()
file2=open("test_clean.txt",'w')
file2.write(test)
file2.close()


