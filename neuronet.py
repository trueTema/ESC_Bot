import kwargs as kwargs
import librosa.display
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self , num_classes = 50):
        super(MyNet , self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 128, kernel_size = 3)
        self.pool1 = nn.MaxPool2d(kernel_size =2 , stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 128 , out_channels = 128 , kernel_size = 3)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout1 = nn.Dropout(p = 0.3)
        self.conv3 = nn.Conv2d(in_channels = 128 , out_channels = 32 , kernel_size = 3)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout2 = nn.Dropout(p = 0.3)
        self.fc1 = nn.Linear(32 * 3 * 25 , 512)
        self.fc2 = nn.Linear(512 , 128)
        self.fc3 = nn.Linear(128, 50)
    def forward(self , x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout2(x)
        #print(x.shape)
        x = x.view(-1 , 32 * 3 *25)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

PATH = "checkpoint.pth"
model = MyNet()

classes = []
class_dict = {}

def load_net():
    model.load_state_dict(torch.load(PATH))
    global classes
    CSV_FILE_PATH = "esc50.csv"
    df = pd.read_csv(CSV_FILE_PATH)
    df.head()
    df = df.drop(['fold', 'esc10', 'src_file', 'take'], axis=1)
    df.head()
    classes = df['category'].unique()
    global class_dict
    class_dict = {i: x for x, i in enumerate(classes)}
    print(class_dict)

predict_res = {'dog': 'лай собак', 'chirping_birds': 'щебетание птиц', 'vacuum_cleaner': 'звук пылесоса',
               'thunderstorm': 'звук грозы', 'door_wood_knock': 'стук в дверь',
               'can_opening': 'звук открытия консервной банки', 'crow': 'ворона', 'clapping': 'хлопки',
               'fireworks': 'звук фейерверка', 'chainsaw': 'звук бензопилы', 'airplane': 'гул самолёта',
               'mouse_click': 'клик мышью', 'pouring_water': 'звук разлития воды', 'train': 'звук поезда',
               'sheep': 'блеянье овцы', 'water_drops': 'звук капель воды', 'church_bells': 'колокольный звон',
               'clock_alarm': 'звук будильника', 'keyboard_typing': 'звуки нажатия по клавиатуре', 'wind': 'гул ветра',
               'footsteps': 'звук шагов', 'frog': 'кваканье лягушки', 'cow': 'корова',
               'brushing_teeth': 'звук чистки зубов', 'car_horn': 'автомобильный гудок',
               'crackling_fire': 'потрескивание огня', 'helicopter': 'звук вертолёта',
               'drinking_sipping': 'пить потягивая',
               'rain': 'звук дождя', 'insects': 'звук насекомых', 'laughing': 'смех', 'hen': 'кудахтанье курицы',
               'engine': 'звук двигателя', 'breathing': 'дыхание', 'crying_baby': 'плач ребёнка',
               'hand_saw': 'звук ручной пилы', 'coughing': 'кашель', 'glass_breaking': 'биение стекла',
               'snoring': 'храп', 'toilet_flush': 'звук смыва в туалете', 'pig': 'хрюкаье свиньи',
               'washing_machine': 'звук стиральной машины', 'clock_tick': 'тик часов', 'sneezing': 'чихание',
               'rooster': 'кудахтанье петуха', 'sea_waves': 'звук волн', 'siren': 'сирена',
               'cat': 'мяуканье кошки', 'door_wood_creaks': 'скрип деревянной двери', 'crickets': 'сверчки'}

def predict(x):
    if x.shape[0] >= 110250:
        size = x.shape[0]
        count = size // 110250
        answers = []
        lists = np.split(x, count)
        for i in range(count):
            list = lists[i]
            list = np.resize(list, (110250))
            mfcc = librosa.feature.mfcc(list, sr = 22050 , n_mfcc = 40)
            mfcc = mfcc.reshape(1 , 40 , 216)
            mfcc = torch.from_numpy(mfcc)
            outputs = model(mfcc)
            max_arg_output = torch.argmax(outputs , dim =1)
            answers.append(max_arg_output)
        result = max(answers , key = answers.count)
        for a, b in enumerate(class_dict):
            if a == result:
                for k, v in predict_res.items():
                    if k == b:
                        return v
    if x.shape[0] < 110250:
        x = np.resize(x , (110250))
        mfcc = librosa.feature.mfcc(x , sr=22050, n_mfcc=40)
        mfcc = mfcc.reshape(1 , 40 , 216)
        mfcc = torch.from_numpy(mfcc)
        outputs = model(mfcc)
        max_arg_output = torch.argmax(outputs , dim =1)
        for a , b in enumerate(class_dict):
            if a == max_arg_output:
                for k, v in predict_res.items():
                    if k == b:
                        return v



# def predict(mfcc):
#     mfcc = np.array(mfcc)
#     mfcc = mfcc.reshape(1, 40, 216)
#     mfcc = torch.from_numpy(mfcc)
#     res_tensor = model(mfcc)
#     cur = torch.argmax(res_tensor, dim=1)
#     for i, j in class_dict.items():
#         if j == int(cur[0]):
#             for k, v in predict_res.items():
#                 if k == i:
#                     return v

#model.eval()

