"""
Brief: We use open-source data set for the simulation environment
@Author: Fei.Liang
@Date: 08/10/2021
Paper proposal:
https://docs.google.com/document/d/1wtd85OB5lnGRIPESCkamNU-O3fMSz_IUwTgspzic8bM/edit#
https://docs.google.com/spreadsheets/d/12XW3PZJMzoQOc3ugGb_I_gqVh8pD2grAcE7QxYFMX0c/edit#gid=0
@Reference:
python_style_rules:
1) https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/#id21
2) https://google.github.io/styleguide/pyguide.html
"""

"""
For action 17, audio + vision, if using duty-cycle, every minute, we trigger the sensors
to collect and send data, 10 hours for standby, it may use extral 144 mAh for data transmitting.
7*124*600/3600 = 144.66666666666666
"""

import constants

MAX_LOCATION_CLASS = 6
PRIVACY_LOCATION_LIST_BASIC = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM]
PRIVACY_LOCATION_LIST_BASIC = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM, constants.LOCATION_LIVINGROOM]
# PRIVACY_LOCATION_LIST_BASIC = [constants.LOCATION_BEDROOM]

PRIVACY_LOCATION_LIST = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM]
# PRIVACY_LOCATION_LIST = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM, constants.LOCATION_KITCHEN]
PRIVACY_LOCATION_LIST = [constants.LOCATION_BEDROOM, constants.LOCATION_BATHROOM, constants.LOCATION_LIVINGROOM]

# PRIVACY_LOCATION_LIST = [constants.LOCATION_BATHROOM]



PRIVACY_ACTIVITY_LIST = [constants.ACTIVITY_MASTER_BEDROOM, constants.ACTIVITY_MASTER_BATHROOM, constants.ACTIVITY_GUEST_BATHROOM]

ROBOT_CANNOT_REG_ACTIVITY_LIST = [constants.ACTIVITY_LEAVE_HOME]