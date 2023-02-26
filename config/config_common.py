class Project:
    name = ''
class COLOR: 
    red = (0, 0, 255)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    fuchsia = (255, 0, 255)
    pink = (241, 167, 228)

class CLASSES:
    g7 = 0
    nabaty = 1
    hop_xanh = 2
    haohao = 3
    mi_do = 4
    nescafe = 5
    mi_vang = 6
    cafe_trang = 7
    

objects =  {
    0:"g7",
    1:"nabaty",
    2:"hop_xanh",
    3:"haohao",
    4:"mi_do",
    5:"nescafe",
    6:"mi_vang",
    7:"cafe_trang"
}

class Visualization:
    allow_visual_area = True
    visual_box_human = True
    visual_pose = False

class AREA:
    '''
    - Area coordinates is the ratio of that point 
    on the image to the real coordinates
    - Coordinate formated ratio: ((tl), (tr), (br), (bl))
    '''
    shelve = ((0.01, 0.14), (0.08, 0.1), (0.29, 0.87), (0.08, 1))
    selection = ((0.12, 0.25), (0.18, 0.25), (0.35, 0.87), (0.3, 0.92))
    # attend = ((0.14, 0.5), (0.4, 0.17), (0.73, 0.33), (0.58, 0.97))
    payment = ((0.48, 0.73), (0.6, 0.63), (0.8, 0.92), (0.55, 1))

    




