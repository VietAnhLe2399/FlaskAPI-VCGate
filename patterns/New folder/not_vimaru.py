with open('other_patterns.txt', encoding='utf-8') as inFile:
    lines = inFile.readlines()

with open('other2_patterns.txt', 'w', encoding='utf-8') as orther:
    with open('vmu_patterns.txt') as outFile:
        lines2 = outFile.readlines()
        # print(lines)
        for l in lines:
            if l not in lines2:
                orther.write(l)