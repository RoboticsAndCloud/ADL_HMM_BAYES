"""
Brief: This code is used to send the request to the game server, do not change the code.

Author: ASCC Lab
Date: 06/01/2022

"""
import json
import requests

# Update the IP Address according the target server
BASE_URL = 'http://10.227.100.46/api/actions/answer/'

# Get the team information 
TEAM_REQ_URL = 'http://10.227.100.46/api/teams/meta/'

DEFAULT_GROUP_NAME = 'ASCCLab'
DEFAULT_GROUP_SCORE = 100

def answer_request(group):
    url = BASE_URL + str(group)
    buzzer_handler(url)

def buzzer_handler(url):
    try:
        r = requests.get(url, verify = False)
        #print(r)
    except Exception as e:
        print("buzzer url error:", e)
    return 0

def get_group_name(group_id):
    url = TEAM_REQ_URL + str(group_id)
    group_name = DEFAULT_GROUP_NAME
    score = DEFAULT_GROUP_SCORE

    try:
        r = requests.get(url, verify = False, timeout=5)
        
        data = json.loads(r.content)
        group_name = data['team']['name']
        score = data['team']['score']
        
    except Exception as e:
        print("buzzer url error:", e)

    return group_name, score

