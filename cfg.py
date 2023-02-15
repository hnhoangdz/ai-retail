shelf_area = [
    [0.2, 0.43], # tl
    [0.39, 0.2], # tr
    [0.44, 0.4], # br
    [0.29, 0.64], # bl
]

attend_area = [
    [0.14, 0.5], # tl
    [0.4, 0.17], # tr
    [0.73, 0.33], # br
    [0.58, 0.97], # bl
]

classes = ["person", "oreo", "coffe", "tuong_ot", "coca", "fanta", "hand"]
area_colors = {
    "outside": (0,0,255),
    "shelf": (0,255,0),
    "attend": (255,0,0),
    "payment": (125,255,0)
}
COUNTER = 0
CONSECUTIVE_FRAME = 5
    