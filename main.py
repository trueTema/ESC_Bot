import os
import queue
from time import time, sleep
import telebot
import pandas as pnd
import cloudconvert
import librosa.display
from threading import Thread
import requests
import subprocess
import matplotlib as plt
import soundfile as sf
import neuronet

token = '5274582075:AAGErCU05dHm9w2MmW_IHOX9SUzINxis4WE'
bot = telebot.TeleBot(token)

cloudconvert.configure(api_key='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiOWIyNGQzOGRkYTI1Y2Q1Mzk3ZjQ3NzhiNzEyNDQzMThiMmM0ZmUxMjFjOWUxMjY4NDEwYjkwNWQ0ODdhMjUwYTRkZjcyYzc3MGMyMWZjODAiLCJpYXQiOjE2NTMzMjc5OTMuNjEyMDEzLCJuYmYiOjE2NTMzMjc5OTMuNjEyMDE0LCJleHAiOjQ4MDkwMDE1OTMuNTk5OTIyLCJzdWIiOiI1ODA2OTQ1MSIsInNjb3BlcyI6W119.VPHSVRM6Rgd_uL6WjIXsMunJMXhb_zIn15pti3xay4QZh6gYH_-F10jsVRpoYLKxnVRP-6B_d5sQiG0tzA_76aEDpOGAICX-1hyEHO5-cUlUlxSWUEyjhrYatu8uVAvj1w8505kr05k_49jFPgG_le_pr7pfhQH2LS6eQ5_5eJISWehiejbGqzXRNaSy9f40NuIyYjbJou0zt850iMuRgotIZwrQPRBIQx52in3O9OgcNqMBNt8a098r0bVPuhiIgNDIrr7jZOM1LOImepPkcSDyn3bDIQPgQBpnnsib1-v4zYmmcqn0Pb-aVx-g7n9tKwnhrzXQagU0ldQsGWQWdN-rU02viZyn8T-oc4Q9wwRGjEQ2gZPvDMrNn53aENtbI-wKnpac5VYUWZaspIPCvJSqDAsvo0rjnS8i7BkWPVb3ZUlk5NzB7lHrmIshybtItJdQwiqaEL9P_uNLODlFbNevqJfNsyz7kdbXLEnBzbh_s_K1934B7Nu3nROM_RGAANDfrU3aiF6mDPfJfyzFR30YRblwAgsfFgdgcgReEOcqstQd9AT6k8R-A3Rw92weWC0eWgOkVx4Dt_BDWHl9YuUgU1YO836eeD2ohbDNq3riXyFNRXsAuhJXzgUwhzYDkKkhbK6XfAtJOtKlo-8te9FoyBI94v3tnu57OlAcVcI', sandbox=False)

neuronet.load_net()

def convertfile(src):

    job = cloudconvert.Job.create(payload={
        "tasks": {
            "import-1": {
                'operation': 'import/upload'
            },
            "task-1": {
                "operation": "convert",
                "input_format": "oga",
                "output_format": "wav",
                "engine": "ffmpeg",
                "input": [
                    "import-1"
                ],
                "audio_codec": "pcm_s16le",
                "audio_bitrate": 128
            },
            "export-1": {
                "operation": "export/url",
                "input": [
                    "task-1"
                ]
            }
        }
    })

    upload_task_id = job['tasks'][0]['id']

    upload_task = cloudconvert.Task.find(id=upload_task_id)
    res = cloudconvert.Task.upload(file_name=src, task=upload_task)
    print(res)
    job = cloudconvert.Job.wait(id=job['id'])
    for task in job["tasks"]:
        if task.get("name") == "export-1" and task.get("status") == "finished":
            export_task = task

            file = export_task.get("result").get("files")[0]
            cloudconvert.download(filename=file['filename'], url=file['url'])
            break

commands = {
    '/help' : 'Полный список команд',
    '/info' : 'Информация о боте',
    '/setname' : 'Установить имя, по которому бот будет обращаться к Вам',
    '/quit' : 'Сбросить профиль в боте'
}
user_list = []
file = pnd.read_csv("users.csv")

class processing_file:
    def __init__(self, x, sr, user_id):
        self.x = x
        self.sr = sr
        self.user_id = user_id



class processing:
    def __init__(self):
        self.q = []

    def push(self, file):
        self.q.append(file)

    def process(self):
        #while True:
            file = self.q[0]
            self.q.pop(0)
            # mfcc = librosa.feature.mfcc(file.x, file.sr, n_mfcc=40)
            # #mfcc.reshape(40, 216)
            # print(mfcc.shape)
            #try:
            x = str(neuronet.predict(file.x))
            bot.send_message(file.user_id, f"Успешно обработано\nПредположительно - {x}")
            #except:
                #bot.send_message(file.user_id, f"Не удалось обработать звук.")





proc = processing()

def process_thr():
    while True:
        if len(proc.q) != 0:
            proc.process()
        sleep(1)

th = Thread(target=process_thr)
th.start()

def remove_row(id):
    df = pnd.read_csv('users.csv')
    for i in range(0,len(df['id'].tolist())):
        if (df['id'].tolist()[i] == id):
            df = df.drop([i])
            df.to_csv('users.csv', index = False)
            return

def write_csv(id, name):
    df = pnd.read_csv('users.csv')
    for i in range(0,len(df['id'].tolist())):
        if (df['id'].tolist()[i] == id):
            df.loc[i] = [id, name]
            df.to_csv('users.csv', index = False)
            return
    df.loc[len(df)] = [id, name]
    df.to_csv('users.csv', index=False)

def find_user(id):
    for i in range(0, len(user_list)):
        if user_list[i].id == id:
            return i
    return -1


class user:
    def __init__(self, id, name):
        self.id = id
        self.isnew = True
        self.name = name
        self.flags = {'getname' : False}
        write_csv(self.id, self.name)

    def __del__(self):
        bot.send_message(self.id, "Вы сбросили свой профиль в боте. \n\nДля начала работы введите /start")

    def getname(self):
        bot.send_message(self.id, "Как я могу к Вам обращаться?")
        self.flags['getname'] = True
        return

    def help(self):
        bot.send_message(self.id, 'Список команд:')
        for i in commands:
            bot.send_message(self.id, i + ' - ' + commands[i])
        return

    def info(self):
        intro = f', {self.name}'
        bot.send_message(self.id, f'Привет{intro}!\n\nЯ - бот, который умеет распознавать звуки окужающей среды)\n\nТы можешь '
                                  'записать мне голосовое сообщение или отправить файл со звуком и я постааюсь '
                                  'распознать, какой именно звук в нём записан. \n\nЭта функция может быть особенно '
                                  'полезна людям с частичной или полной потерей слуха.')
        return


    def quit(self):
        remove_row(self.id)
        user_list.pop(find_user(self.id))

    def start(self):
        if self.isnew == True:
            self.info()
            self.getname()
        else:
            bot.send_message(self.id, f"С возвращением, {self.name}!")
        return

    def cmd_hand(self, cmd):
        if (cmd == "/quit"):
            self.quit()
            return
        if (cmd == "/start"):
            self.start()
            return
        if (cmd == "/help"):
            self.help()
            return
        if (cmd == "/info"):
            self.info()
            return
        if (cmd == "/setname"):
            self.getname()
            return
        return

    def txt_hand(self, name):
        if (self.flags['getname'] == True):
            self.name = name
            write_csv(self.id, self.name)
            self.flags['getname'] = False
            if self.isnew == True:
                bot.send_message(self.id, f"Добро пожаловать, {self.name}!")
                self.isnew = False
            else:
                bot.send_message(self.id, f"Теперь я знаю вас под другим именем, {self.name}!")
            return
        return

    def doc_hand(self, src):
        allowed_ext = ['wav', 'ogg', 'mp3']
        if (src[-3:] in allowed_ext):
            x, sr = librosa.load(src)
            proc.push(processing_file(x, sr, self.id))
            os.remove(src)
            return
        else:
            bot.send_message(self.id, f"Недопустимый формат файла. (Допустимые расширения - {allowed_ext}")
            os.remove(src)
            return

users = file['id'].tolist()
for i in range(0, len(users)):
    nm = str(file['name'][i])
    # if str(file['name'][i]) == 'nan':
    #     nm = 'незнакомец'
    print(nm, file['name'][i], users[i])
    user_list.append(user(users[i], nm))
    user_list[i].isnew = False
    print(user_list[i].id, user_list[i].name)


@bot.message_handler(commands=["start", "help", "info", "setname", "quit"])
def get_command(message):
    id = message.from_user.id
    res = find_user(id)
    if (res != -1):
        user_list[res].cmd_hand(message.text)
    else:
        user_list.append(user(id, 'незнакомец'))
        user_list[len(user_list) - 1].cmd_hand(message.text)

@bot.message_handler(content_types=["text"])
def get_text(message):
    id = message.from_user.id

    res = find_user(id)
    if (res != -1):
        user_list[res].txt_hand(message.text)
    else:
        user_list.append(user(id, 'незнакомец'))
        user_list[len(user_list) - 1].txt_hand(message.text)

@bot.message_handler(content_types=["document"])
def get_audio(message):
    id = message.from_user.id
    file_info = bot.get_file(message.document.file_id)
    file = bot.download_file(file_info.file_path)
    pos = file_info.file_path.find('/') + 1
    src = os.path.join(f'{file_info.file_path[pos:]}')
    with open(src, 'wb') as f:
         f.write(file)
    res = find_user(id)
    if (res != -1):
        user_list[res].doc_hand(src)
    else:
        user_list.append(user(id, 'незнакомец'))
        user_list[len(user_list) - 1].doc_hand(src)


@bot.message_handler(content_types=["voice"])
def get_voice(message):
    id = message.from_user.id

    file_info = bot.get_file(message.voice.file_id)
    file = bot.download_file(file_info.file_path)
    path = os.path.join(f'{message.chat.id}_{int(time())}.wav')
    src = os.path.join(f'{message.chat.id}_{int(time())}.oga')
    with open(src, 'wb') as f:
        f.write(file)
    convertfile(src)
    os.remove(src)
    res = find_user(id)
    if (res != -1):
        user_list[res].doc_hand(path)
    else:
        user_list.append(user(id, 'незнакомец'))
        user_list[len(user_list) - 1].doc_hand(path)


bot.polling(non_stop=True, interval = 0)