# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import random
from torch.autograd import Variable
import numpy as np
from FXHIST.DatasetFXHIST import DatasetFXHIST, FXHISTDataLoader

'''
******************************
Инициализация Гипер-параметров
******************************
'''
input_size = 720
# Количество узлов на скрытых слоях:
hidden_size = {0: input_size,
               1: 660,
               2: 330,
               3: 111}
classes ={-1:'SELL',0:'LOSS',1:'BUY'}
num_classes = len(classes)  # Число классов на выходе. В этом случае 3: от -1 до 1
num_epochs = 50  # Количество тренировок всего набора данных
batch_size = 1  # Размер входных данных для одной итерации
learning_rate = 0.0001  # Скорость конвергенции

'''
**********************************
Создаем нейронную сеть Feedforward
**********************************
'''
class Net(nn.Module):
    '''
    @property
    def input_size(self):
        return self.input_size

    @property
    def hiden_size(self):
        return self.hidden_size

    @property
    def num_classes(self):
        return self.num_classes
    '''
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()  # Наследуемый родительским классом nn.Module

        self.fc1 = nn.Linear(input_size, hidden_size[1])  # 1й связанный слой:
        # 660 (данные входа) -> 660 (скрытый узел)
        self.relu1 = nn.ReLU()  # Нелинейный слой ReLU max(0,x)

        self.fc2 = nn.Linear(hidden_size[1], hidden_size[2])  # 2й связанный слой:
        # 660 (1 скрытый узел) -> 330 (2 скрытый узел)
        self.relu2 = nn.ReLU()  # Нелинейный слой ReLU max(0,x)

        self.fc3 = nn.Linear(hidden_size[2], hidden_size[3])  # 3й связанный слой:
        # 330 (2 скрытый узел) -> 111 (3 скрытый узел)
        self.relu3 = nn.ReLU()  # Нелинейный слой ReLU max(0,x)

        self.fc4 = nn.Linear(hidden_size[3], num_classes)  # 4й связанный слой:
        # 111 (3 скрытый узел) -> 3 (класс вывода)

        self.input_size = input_size
        self.hiden_size = hidden_size
        self.num_classes = num_classes

    def forward(self, x):  # Передний пропуск: складывание каждого слоя вместе
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out


# **********************************
# Демонстрация нейросети
# **********************************
net = Net(input_size, hidden_size, num_classes)
print(net)

'''
inp = torch.from_numpy(np.array([random.random() for i in range(input_size)]))
inp = inp.double()
inp = Variable(inp)
ask=net(inp)

print(f'Тестовый запуск сети:\nask=net(<inp>)={ask}')
'''


'''Выбираем функцию потерь и оптимизатор'''
# Осуществляем оптимизацию путем стохастического градиентного спуска
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# Создаем функцию потерь
criterion = nn.NLLLoss()
if __name__ == '__main__':

    trane_minutes = DatasetFXHIST()

    print(trane_minutes.data.head(15))

    trane_loader = FXHISTDataLoader(
        dataset=trane_minutes,
        batch_size=batch_size)


    # '''
    # ********
    # ПРОВЕРКА
    # ********
    # '''
    # for i, (q, p) in enumerate(trane_loader):
    #     try:
    #         q = Variable(q.view(-1,input_size))
    #         if q.shape[0] == 0:
    #             continue
    #     except:
    #         q = Variable(q.view(-1))
    #         continue
    #     p = Variable(p)
    #
    #
    #     print(f'{i} Variables:\n{p},\n[{q.shape}],\n--------------------')
    # print(trane_loader)

    '''
    *******************
    Тренируем нейросеть
    *******************
    '''
    for epoch in range(num_epochs):
        for i, (q, p) in enumerate(trane_loader):
            try:
                q = Variable(q.view(-1, input_size))
                if q.shape[0] == 0:
                    continue
            except:
                q = Variable(q.view(-1))
                continue
            p = Variable(p.view(-1))

            print(f'{i} Variables:\n{p},\n[{q.shape}],\n--------------------')

            optimizer.zero_grad()  # Инициализация скрытых масс до нулей
            outputs = net(q)  # Передний пропуск: определение выходного класса, данного изображения
            loss = criterion(outputs,
                             p)  # Определение потерь:
            # разница между выходным классом и предварительно заданной меткой
            loss.backward()  # Обратный проход:
            # определение параметра weight
            optimizer.step()  # Оптимизатор:
            # обновление параметров веса в скрытых узлах

            if (i + 1) % 200 == 0:  # Логирование
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f' %
                      (epoch + 1, num_epochs, i + 1, len(trane_minutes) // batch_size, loss.item()))

'''
if __name__ == '__main__':
    m_dataset = DatasetFXHIST('data/eurusdhist/eurusd_minute.csv')

    print(m_dataset.data.head())
    print(f'There are {m_dataset.nRow} rows and {m_dataset.nCol} columns')

    print('-------------')
    p = list()
    for i in range(0, m_dataset.nRow, 15):
        itm = m_dataset.__getitem__(i)
        if itm[1] is not None:
            print(i, '\t>\n', list(itm))
'''
