# Reference: https://python-socketio.readthedocs.io/en/latest/intro.html

import asyncio
import socketio
import time

DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME = 'DATA_FILE_RECEIVED_FROM_WMU'

DATA_RECOGNITION_FROM_WMU_EVENT_NAME = 'DATA_RECOGNITION_FROM_WMU'

DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME = 'DATA_RECOGNITION_TO_ADL'

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('connection established')

@sio.event
async def my_message(data):
    print('message received with ', data)
    await sio.emit('my response', {'response': 'my response'})

@sio.event
async def disconnect():
    print('disconnected from server')


@sio.on(DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME)
async def on_message(data):
    print('Got new data:', data)
    time.sleep(2)
    event_name = DATA_RECOGNITION_FROM_WMU_EVENT_NAME
    broadcasted_data = {'data': "test message!"}
    await sio.emit(event_name, broadcasted_data)
    print('send recognition :', data)


@sio.on(DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME)
async def on_message(data):
    print('Got final recognition data:', data)


async def main():
    await sio.connect('http://127.0.0.1:5000')
    await sio.wait()

if __name__ == '__main__':
    asyncio.run(main())
