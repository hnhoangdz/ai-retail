shelf_area = [
    [0.2 , 0.43], # tl
    [0.4 , 0.17], # tr
    [0.5 , 0.39], # br
    [0.33, 0.68] # bl
]

attend_area = [
    [0.14, 0.5], # tl
    [0.4 , 0.17], # tr
    [0.73, 0.33], # br
    [0.58, 0.97] # bl
]

classes = ["person", "oreo", "coffe", "tuong_ot", "coca", "fanta", "hand"]
area_colors = {
    "outside": (0,0,255),
    "shelf": (0,255,0),
    "attend": (255,0,0),
    "payment": (125,255,0)
}

COUNTER = 0
CONSECUTIVE_FRAME = 10

class COLOR:
    red = (0,0,255)
    blue = (255,0,0)
    green = (0,255,0)
    yellow = (255,255,0)
    magenta = (255,0,255)

classes = {
    0:"person",
    1:"oreo",
    2:"coffe",
    3:"tuong_ot",
    4:"coca",
    5:"fanta",
    6:"hand",
}

    