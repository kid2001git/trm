

# strange sentence ex. too short, delimiter 2 above, ... 
fp1 = open("../../spm-data/train.txt", 'r', encoding='utf-8')
i=0
while True:
    i = i+1
    line = fp1.readline()
    # if line!='\n' and len(line)<5: # too short
    #     print(line, '%d'%i)
    if line.find('&&&') != line.rfind('&&&'):
        print(line, '%d'%i )

    if not line:
        break
fp1.close()

