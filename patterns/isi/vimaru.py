list = ['VMU', 'vmu', 'am Maritime Uni', 'am Maritime uni', 'am maritime uni', 'am maritime uni']

with open('other_patterns.txt', encoding='utf-8') as inFile:
    with open('vmu_patterns.txt', 'w') as outFile:
        for l in inFile.readlines():
            for j in list:
                if j in l:
                    outFile.write(l)
                    break