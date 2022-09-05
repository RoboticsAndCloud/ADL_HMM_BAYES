"""
Brief: This code is used to display the group name and score on the LCD screen.

Author: ASCC Lab
Date: 06/01/2022

Reference: pip3 install python-socketio 
           pip3 install aiohttp
Run: python3 Ex3_lcd_score.py

"""
import client_lib
import asyncio
import signal
import socketio
import functools
import time


# Update the IP Address according the target server
IP_ADDRESS = 'http://127.0.0.1:5000'
# Update your group ID
GROUP_ID = 1

INTERVAL = 10

shutdown = False


DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME = 'DATA_FILE_RECEIVED_FROM_WMU'
DATA_RECOGNITION_FROM_WMU_EVENT_NAME = 'DATA_RECOGNITION_FROM_WMU'

DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME = 'DATA_RECOGNITION_TO_ADL'

DATA_TYPE = 'type'
DATA_CURRENT = 'current_time'
DATA_FILE = 'file'
DATA_TYPE_IMAGE = 'image'
DATA_TYPE_SOUND = 'audio'
DATA_TYPE_MOTION = 'motion'



# For getting the score
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')

@sio.on(DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME)
async def on_message(data):
    print('Got new data:', data)
    try:
        if data[DATA_TYPE] == DATA_TYPE_MOTION:
            print('Get motion:', data)

        cur_time = data[DATA_CURRENT]
        file = data[DATA_FILE]
        print('cur_time:', cur_time, 'file:', file)
    except:
        return
        pass
    time.sleep(2)
    event_name = DATA_RECOGNITION_FROM_WMU_EVENT_NAME
    broadcasted_data = {'data': "test message!"}
    await sio.emit(event_name, broadcasted_data)
    print('send recognition :', data)


@sio.on(DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME)
async def on_message(data):
    try:
        if data['type'] == DATA_TYPE_IMAGE:
            print('Get image:', data)
    except:
        pass
    print('Got final recognition data:', data)


@sio.event
async def disconnect():
    print('disconnected from server')

def stop(signame, loop):
    global shutdown
    shutdown = True

    tasks = asyncio.all_tasks()
    for _task in tasks:
        _task.cancel()

async def run():
    cnt = 0
    global shutdown
    while not shutdown:
        print('.', end='', flush=True)

        try:
            await asyncio.sleep(INTERVAL)
            cnt = cnt + INTERVAL
            print('run: ', cnt)
            # event_name = DATA_RECOGNITION_FROM_WMU_EVENT_NAME
            # broadcasted_data = {'type': DATA_TYPE_IMAGE, 'file': 'image0'}
            # await sio.emit(event_name, broadcasted_data)
        except asyncio.CancelledError as e:
            pass
            #print('run', 'CancelledError', flush=True)

    await sio.disconnect()

async def main():
    await sio.connect(IP_ADDRESS)

    loop = asyncio.get_running_loop()

    for signame in {'SIGINT', 'SIGTERM'}:
        loop.add_signal_handler(
            getattr(signal, signame),
            functools.partial(stop, signame, loop))

    task = asyncio.create_task(run())
    try:
        await asyncio.gather(task)
    except asyncio.CancelledError as e:
        pass
        #print('main', 'cancelledError')

    print('main-END')


if __name__ == '__main__':
    asyncio.run(main())

