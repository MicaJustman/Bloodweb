from pynput.mouse import Button, Controller
from pynput import keyboard
from imagehash import hex_to_hash, colorhash, hex_to_flathash, average_hash
from win32api import RGB
from win32gui import IsWindowVisible, GetWindowText, GetWindowDC, ReleaseDC, DeleteObject, EnumWindows, GetWindowRect, \
    SetWindowLong, GetWindowLong, SetLayeredWindowAttributes, SetWindowPos
from win32ui import CreateDCFromHandle, CreateBitmap
from win32con import SRCCOPY, GWL_EXSTYLE, WS_EX_LAYERED, HWND_TOP, LWA_COLORKEY, SWP_NOSIZE
from numpy import asarray
from PIL.Image import frombuffer, fromarray
from PIL.ImageDraw import Draw
from time import sleep
import pygame
import json


DBDhwnd = None
mode = 0
page = 2
part = 3
mouse = Controller()
halt = False

AdjecentDict={
    1 : [18, 7, 8, 9],
    2 : [8, 9, 10, 11],
    3 : [10, 11, 12, 13],
    4 : [12, 13, 14, 15],
    5 : [14, 15, 16, 17],
    6 : [16, 17, 18, 7],
    7 : [19, 20],
    8 : [20, 21],
    9 : [21, 22],
    10 : [22, 23],
    11 : [23, 24],
    12 : [24, 25],
    13 : [25, 26],
    14 : [26, 27],
    15 : [27, 28],
    16 : [28, 29],
    17 : [29, 30],
    18 : [30, 19]
}

webIndex=[(610, 469), (705, 523), (706, 634), (611, 690), (516, 634), (516, 523), (555, 367), (666, 368), (771, 424), (828, 524),
           (828, 634), (771, 733), (667, 789), (555, 790), (450, 734), (394, 633), (393, 523), (450, 424), (449, 304), (611, 260),
           (773, 304), (892, 421), (935, 578), (891, 736), (774, 853), (611, 896), (449, 854), (331, 734), (286, 578), (331, 422)]

itemIndex = [(356, 580), (458, 581), (559, 581), (660, 581), (761, 581),
             (357, 680), (457, 680), (558, 680), (660, 680), (761, 680),
             (356, 779), (457, 779), (559, 779), (660, 779), (761, 779)]

def on_press(key):
    global halt
    halt = True

def winEnumHandler( hwnd, ctx ):
    if IsWindowVisible( hwnd ):
        if str(GetWindowText(hwnd)) == 'DeadByDaylight  ':
            global DBDhwnd
            DBDhwnd = hwnd

def grabImage(width, height, offset_width, offset_height, DBDhwnd):
    wDC = GetWindowDC(DBDhwnd)
    dcObj = CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (width, height), dcObj, (offset_width, offset_height), SRCCOPY)

    bmpinfo = dataBitMap.GetInfo()
    bmpstr = dataBitMap.GetBitmapBits(True)
    im = frombuffer('RGBA', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'RGBA', 0, 1)
    img = asarray(im)
    img = img[:, :, [2, 1, 0]]

    dcObj.DeleteDC()
    cDC.DeleteDC()
    ReleaseDC(DBDhwnd, wDC)
    DeleteObject(dataBitMap.GetHandle())
    return img

def recordItems(key):
    global itemIndex
    itemIndex.append((mouse.position[0], mouse.position[1]))
    if len(itemIndex) == 15:
        print(itemIndex)

def recordWeb(key):
    global webIndex
    webIndex.append((mouse.position[0], mouse.position[1]))
    if len(webIndex) == 30:
        print(webIndex)

EnumWindows( winEnumHandler, None )
size = GetWindowRect(DBDhwnd)
width = size[2]
height = size[3]

if mode == 0:
    listener = keyboard.Listener(on_press=on_press)
    print("Press any key to stop")
    listener.start()

    '''for x in webIndex:
        if halt:
            exit()
    
        mouse.position = x
        mouse.press(Button.left)
        sleep(.5)
        mouse.release(Button.left)'''

    img = grabImage(width, height, 0, 0, DBDhwnd)
    fromarray(img).save("test.png")

    counter = 1

    images = []
    for x in webIndex:
        temp = img[x[1] - 22:x[1] + 22, x[0] - 22:x[0] + 22]
        fromarray(temp).save("pics/" + str(counter) + ".png")
        images.append(temp)
        counter += 1

    file = open("ItemRecords.json", "r")
    Records = json.load(file)

    feature = abs(average_hash(fromarray(images[11])) - hex_to_hash(Records["SURVIVOR"]["17"][0]))

    print(Records["SURVIVOR"]["17"][1])

    print(feature)


elif mode == 1:
    #show web coords

    # uncomment this line and clear webIndex to record new indexes. Hit any key to store the location of the mouse
    '''with keyboard.Listener(on_press=recordweb) as listener:
            listener.join()'''

    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    done = False
    fuchsia = (255, 0, 128)  # Transparency color
    blue = (0, 0, 255)

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*fuchsia), 0, LWA_COLORKEY)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(fuchsia)  # Transparent background

        for x in webIndex:
            pygame.draw.circle(screen, blue, (x[0], x[1]), 36)


        pygame.display.update()

elif mode == 2:
    # show inventory coords

    #uncomment this line and clear itemIndex to record new indexes. Hit any key to store the location of the mouse
    '''with keyboard.Listener(on_press=recordItems) as listener:
        listener.join()'''

    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    done = False
    fuchsia = (255, 0, 128)  # Transparency color
    blue = (0, 0, 255)

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*fuchsia), 0, LWA_COLORKEY)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(fuchsia)  # Transparent background

        for x in itemIndex:
            pygame.draw.rect(screen, blue, pygame.Rect(x[0], x[1], 72, 72))

        pygame.display.update()

elif mode == 3:
    loadChar = "SURVIVOR"


    #opens the json, reads into a var, and truncates
    file = open("ItemRecords.json", "r+")
    Records = json.load(file)
    file.seek(0)
    file.truncate()

    #for population ItemRecords
    img = grabImage(width, height, 0, 0, DBDhwnd)
    fromarray(img).save("test.png")

    #calculates hash for each image and appends to dict
    counter = (page - 1) * 15 + 1
    temp = 0
    for x in range(part):
        if str(counter) in Records[loadChar]:
            temp = Records[loadChar][str(counter)][1]
        else:
            temp = ""
        temp2 = img[itemIndex[x][1]:itemIndex[x][1] + 74, itemIndex[x][0]:itemIndex[x][0] + 74]
        Records[loadChar][str(counter)] = [str(average_hash(fromarray(temp2))), temp]
        counter += 1

    json = json.dumps(Records, indent=5)
    file.write(json)
    file.close()