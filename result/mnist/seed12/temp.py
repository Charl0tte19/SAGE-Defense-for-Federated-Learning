seed = 'seed_12'

s = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/poison.txt', 'r')
r = s.readlines()

localfile = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/local.txt', 'r')
localfile = localfile.readlines()

attackfile = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/attacker.txt', 'r')
attackfile = attackfile.readlines()

print(len(attackfile))
attacker = (attackfile[1][15:])

com = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/cur.txt', 'r')
comm = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/com.txt', 'w')

com = com.readlines()
com[0] = com[0][:-1]
com[1] = com[1][:-1]
#print((localfile))
'''
local = []
for i in range(len(localfile)):
	local.append(localfile[i][10:-2])
'''
#print((local))


ingood = r[len(r) - 5]
ingood = ingood[ingood.find(': ')+3:-2]
ingood = ingood.split(', ')

for i in range(len(ingood)):
	ingood[i] = int(ingood[i])
#print(ingood)

local = []
local2 = []
for i in range(len(localfile)):
	local2.append(localfile[i][10:-2])
rond = ''
local = [x for _,x in sorted (zip(ingood, local2))]
#print(len(r))
for i in range(len(r)-1,0,-1):
#	print(i)
	if r[i].find('=== Round  ') != -1:
		zzz = r[i].find('=== Round  ')+10
		rond = (r[i][zzz:zzz+3])
		break
best_round = eval(r[-15].split(" ")[-1])
best_round = " " + str(best_round)
alll = [[],[]]
print(best_round)
for i in range(len(r)):
	if r[i].find('=== Round') != -1:
		if r[i].find(best_round) != -1:
			alll[0].append(int(r[i-3][6:-1]))
			alll[1].append(r[i+3][r[i+3].find(' Testing accuracy: ')+19:r[i+3].find(' loss:')])



good = [[],[]]
print(alll)
for i in range(len(alll[0])):
        for j in range(len(ingood)):
                if alll[0][i] == ingood[j]:
                        good[0].append(ingood[j])
                        good[1].append(alll[1][i])

print(good)

z = [x for _,x in sorted (zip(good[1], good[0]), reverse=True)]
ingoodorder = [x for _,x in sorted (zip(good[1], local), reverse=True)]
#print('===================================')
#print(ingoodorder)
#print('===================================')
#print(z)
t = (sorted(good[1], reverse=True))
first3=[[],[]]
first3[0] = z[0:3]
first3[1] = t[0:3]
print(first3)
#print(local)
localout = '['
localout += (ingoodorder[0]) + ', '
localout += (ingoodorder[1]) + ', '
localout += (ingoodorder[2]) + ']'
print(localout)
#print(ingoodorder)

''' 
for i in range(len(alll)):
	result = (min(alll[1]))
	alll[1].find(result)
'''
#print(alll[0])
#print(alll[1]
rond = int(rond)
c = 'python -u after_preprocess_FL_mnist.py --seed ' + seed[-2:] + ' --epoch ' + str(99-rond) + ' --noniid ' + com[0] + ' --attack_ratio ' + com[1] + ' --test_label_acc --target_random --model_path=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/final.pt --pretrained_model=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/poison_' + com[1] + '_notScale_0.pt(' + str(first3[0][0]) + ').pt1 --local_file=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/local.txt --attacker_file=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/attacker.txt'
c2 = 'python -u after_preprocess_FL_mnist.py --seed ' + seed[-2:] + ' --epoch ' + str(99-rond) + ' --noniid ' + com[0] + ' --attack_ratio ' + com[1] + ' --test_label_acc --target_random --scale --model_path=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/final.pt --pretrained_model=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/poison_' + com[1] + '_Scale_0.pt(' + str(first3[0][0]) + ').pt1 --local_file=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/local.txt --attacker_file=./mnist/' + seed + '/noniid_' + com[0] + '/ratio_' + com[1] + '/attacker.txt'

if float(com[1]) < 0.1:
	comm.write(c2) 
else:
	comm.write(c)


localo = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/localout.txt', 'w')

attacko = open('/home/ubuntu/My_FL/code/mnist/' + seed + '/attackerout.txt', 'w')

attacko.write(attacker)
localo.write(localout)
attacko.close()
localo.close()
#localout
#first3


s.close()
