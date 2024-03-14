source = "./Weekly/2023-03-14-average_sgd_gravity.txt"
bracket = "./bracket0_dif.txt"

with open(source, "r") as doc:
    src = doc.read().split("RANKINGS")[-1]
with open(bracket, "r") as doc:
    br = eval(doc.read())

lookup = {k : v for k, v in eval(src)}

br = [i + (lookup[i[1]], 0) for i in br]

def populate_bracket(r1):
    rounds = [r1]
    remaining = r1
    victors = []
    for i in range(int(len(remaining)/2)):
        result = remaining[i * 2] if remaining[i * 2][2] > remaining[(i * 2) + 1][2] else remaining[(i * 2) + 1]
        loser = remaining[(i * 2) + 1] if remaining[i * 2][2] > remaining[(i * 2) + 1][2] else remaining[i * 2]
        result = result[:-1] + (result[-1] + loser[-2] + loser[-1],)
        victors.append(result)
    rounds.append(victors)
    remaining = victors
    while len(remaining) > 1:
        victors = []
        for i in range(int(len(remaining)/2)):
            result = remaining[i * 2] if remaining[i * 2][-1] < remaining[(i * 2) + 1][-1] else remaining[(i * 2) + 1]
            loser = remaining[(i * 2) + 1] if remaining[i * 2][-1] < remaining[(i * 2) + 1][-1] else remaining[i * 2]
            result = result[:-1] + (result[-1] + loser[-2] + loser[-1],)
            victors.append(result)
        remaining = victors
        rounds.append(victors)
    return rounds

if __name__ == "__main__":
    print([[g[:2] for g in i] for i in populate_bracket(br)][1:])