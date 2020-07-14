list = ['VMU', 'vmu', 'Vietnam Maritime Uni', 'Vietnam Maritime uni', 'Viet Nam Maritime Uni', 'Viet Nam Maritime uni', 'VietNam Maritime Uni', 'maritime']

with open('other_patterns.txt', encoding='utf-8') as inFile:
    with open('vmu_patterns.txt', 'w') as outFile:
        for l in inFile.readlines():
            for j in list:
                if j in l:
                    outFile.write(l)
                    break