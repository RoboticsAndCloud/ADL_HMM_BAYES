#WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead

#To Solve WARNING: Here you should use Waitress a production WSGI server. If you are deploying your application to production then you have to use waitress. Follow this simple example.
# https://exerror.com/warning-this-is-a-development-server-do-not-use-it-in-a-production-deployment-use-a-production-wsgi-server-instead/

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME = 'DATA_FILE_RECEIVED_FROM_WMU'
DATA_RECOGNITION_FROM_WMU_EVENT_NAME = 'DATA_RECOGNITION_FROM_WMU'

DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME = 'DATA_RECOGNITION_TO_ADL'


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

socketio = SocketIO()
socketio.init_app(app, cors_allowed_origins='*')

name_space = ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/push')
def push_once():
    event_name = DATA_FILE_RECEIVED_FROM_WMU_EVENT_NAME
    broadcasted_data = {'data': "test message!"}
    socketio.emit(event_name, broadcasted_data, namespace=name_space)
    return 'done!'

@socketio.on(DATA_RECOGNITION_FROM_WMU_EVENT_NAME, namespace=name_space)
def on_msg_recognition_result(data):
    # send the notice to the main to get the recognition results
    event_name = DATA_RECOGNITION_FINAL_TO_ADL_EVENT_NAME
    broadcasted_data = data
    socketio.emit(event_name, broadcasted_data, namespace=name_space)
    print('Got recognition result:', data)
    return 'done!'



@socketio.on('connect', namespace=name_space)
def connected_msg():
    print('client connected.')

@socketio.on('disconnect', namespace=name_space)
def disconnect_msg():
    print('client disconnected.')

@socketio.on('my_event', namespace=name_space)
def mtest_message(message):
    print(message)
    emit('my_response',
         {'data': message['data'], 'count': 1})

if __name__ == '__main__':

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)






