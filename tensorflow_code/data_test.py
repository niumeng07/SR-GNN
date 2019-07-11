import pickle

train_data = pickle.load(open('../datasets/sample/train.txt', 'rb'))

print(len(train_data[0])) #train_data[0]:inputs->sequences
print(len(train_data[1])) #train_data[1]:targets->targets

for i in range(0, 10):
    print(train_data[0][i], train_data[1][i])
