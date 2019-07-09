# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

'''Инициализация Гипер-параметров'''
#Гипер-параметры – это мощные аргументы с предварительной настройкой
# и не будут обновляться в ходе изучения нейронной сети.
input_size = 784       # Размеры изображения = 28 x 28 = 784
hidden_size = 500      # Количество узлов на скрытом слое
num_classes = 10       # Число классов на выходе. В этом случае от 0 до 9
num_epochs = 50        # Количество тренировок всего набора данных
batch_size = 150       # Размер входных данных для одной итерации
learning_rate = 0.0001  # Скорость конвергенции

'''Загрузка набора данных MNIST'''
#MNIST – это огромная база данных с тоннами прописанных чисел (т.е. от 0 до 9),
# которая направлена на обработку изображений.
train_dataset = dsets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = dsets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor()
)
#Загрузка набора данных.
# После загрузки MNIST, мы загружаем набор данных в наш код:
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

print(test_dataset.data)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)
#Обратите внимание:
# мы перемешиваем процесс загрузки train_dataset, чтобы процесс обучения не зависел от порядка данных,
# однако порядок test_loader остается неизменным, чтобы понять, когда мы можем обработать неопределенный порядок входов.

'''Создаем нейронную сеть Feedforward'''
#Структура модели нейросети
#
# Нейронная сеть включает в себя два полностью соединенных слоя (т.е. fc1 и fc2) и нелинейный слой ReLU между ними.
# Как правило, мы называем эту структуру 1-скрытый слой нейросети, отбрасывая слой вывода (fc2).
#
# Запустив следующий код, указанные изображения (х) могут пройти через нейронную сеть и сгенерировать вывод (out),
# показывая, как именно соответствие принадлежит каждому из 10 классов.
# Например, изображение кошки соответствует изображению собаки на 0.8,
# в то врем я как соответствие изображению самолета – 0.3.
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()  # Наследуемый родительским классом nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1й связанный слой: 784 (данные входа) -> 500 (скрытый узел)
        self.relu = nn.ReLU()  # Нелинейный слой ReLU max(0,x)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.relu6 = nn.ReLU6()
        self.fc3 = nn.Linear(hidden_size, num_classes)  # 2й связанный слой: 500 (скрытый узел) -> 10 (класс вывода)

    def forward(self, x):  # Передний пропуск: складывание каждого слоя вместе
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu6(out)
        out = self.fc3(out)
        return out

#Демонстрация нейросети
# Мы только что создали настоящую нейронную сеть по нашей структуре.
net = Net(input_size, hidden_size, num_classes)
print(net)

#Включаем графический процессор (GPU)
# Обратите внимание: вы можете включить эту строку для запуска кодов на GPU
# net.cuda() # Вы можете прокомментировать эту строку для отключения GPU

'''Выбираем функцию потерь и оптимизатор'''
#Функция потерь (критерий) выбирает, как выходные данные могут быть сопоставлены с классом.
# Это определяет, как хорошо или плохо работает нейросеть.
# Оптимизатор выбирает способ обновления веса, чтобы найти область,
# в которой будет найден лучшие параметры в конкретной нейросети.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

'''Тренируем нейросеть'''
#Этот процесс займет примерно 3-5 минут, в зависимости от работоспособности вашего компьютера.
# Подробные инструкции находятся в комментариях (после #) в следующих примерах.
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными, классом
        images = Variable(images.view(-1, 28 * 28))  # Конвертация тензора в переменную:
        # изменяем изображение с вектора, размером 784 на матрицу 28 x 28
        labels = Variable(labels)

        optimizer.zero_grad()  # Инициализация скрытых масс до нулей
        outputs = net(images)  # Передний пропуск: определение выходного класса, данного изображения
        loss = criterion(outputs,
                         labels)  # Определение потерь:
        # разница между выходным классом и предварительно заданной меткой
        loss.backward()  # Обратный проход:
        # определение параметра weight
        optimizer.step()  # Оптимизатор:
        # обновление параметров веса в скрытых узлах

        if (i + 1) % 200 == 0:  # Логирование
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.8f' %
                  (epoch + 1,num_epochs,i + 1,len(train_dataset) // batch_size,loss.item()))

'''Тестируем модель нейросети'''
#Также как и с тренировкой нейронной сети,
# нам также нужно загрузить пачки тестируемых изображений и собрать выходные данные.
# Отличия теста от тренировки:
#
# 1.Проходит без подсчета потерь и веса;
# 2.Нет обновления веса;
# 3.Корректный расчет прогноза
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)  # Выбор лучшего класса из выходных данных:
    # класс с лучшим счетом
    total += labels.size(0)  # Увеличиваем суммарный счет
    correct += (predicted == labels).sum()  # Увеличиваем корректный счет

print('Accuracy of the network on the 10K test images: %d %%' % (100 * correct / total))
'''Мы сохраняем тренированную модель как pickle. Таким образом, ее можно будет загрузить и использовать в будущем.'''
torch.save(net.state_dict(), 'fnn_model.pkl')

print(net.state_dict())