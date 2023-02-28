# shelf_area = [
#     [0.2 , 0.43], # tl
#     [0.4 , 0.17], # tr
#     [0.5 , 0.39], # br
#     [0.33, 0.68] # bl
# ]

# attend_area = [
#     [0.14, 0.5], # tl
#     [0.4 , 0.17], # tr
#     [0.73, 0.33], # br
#     [0.58, 0.97] # bl
# ]

in_out_area = [
    [0.8, 0.64],
    [0.95, 0.98]
]

shelf_area = [
    [0.4, 0.124],
    [0.5, 0.57]
]

payment_area = [
    [0.6, 0.67],
    [0.75, 0.637],
    [0.827, 0.863],
    [0.684, 0.94]
]

area_colors = {
    "outside": (0,0,255),
    "shelf": (0,255,0),
    "attend": (255,0,0),
    "payment": (125,255,0)
}

COUNTER = 0
CONSECUTIVE_FRAME = 3

class COLOR:
    red = (0,0,255)
    blue = (255,0,0)
    green = (0,255,0)
    yellow = (255,255,0)
    magenta = (255,0,255)

classes = ['dark_noodles', 'g7', 'hand', 'haohao', 'modern', 'nabaty', 'nescafe', 'oreo', 'passiona']

# import cv2
# from PIL import ImageFont

# TITLE = "---------Receipt----------"
# THICKNESS_TITLE = 2
# FONT_TITLE = cv2.FONT_HERSHEY_TRIPLEX
# COLOR_TITLE = (111, 0, 5)
# FONT_SCALE=0.7

# GREEN = (0,255,0)
# BLACK = (0,0,0)
# FONT = cv2.FONT_HERSHEY_COMPLEX
# THICKNESS = 1
# LINE_SPACE = 30

# ORG_STT = (10,60)
# X_STT = 10

# ORG_RETAIL = (70, 60)
# X_RETAIL = 70

# ORG_AMOUNT = (260, 60)
# X_AMOUNT = 260

# ORG_PRICE = (380, 60)
# X_PRICE = 380

# ORG_TOTAL = (500, 60)
# X_TOTAL = 500

# ORG_ALL = (530, 60)
# X_ALL = 540

# IMAGE_SIZE = 640
# CONF_THRESH = 0.75
# IOU_THRESH = 0.45



    