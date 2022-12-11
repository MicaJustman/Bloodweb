import os
from time import sleep

import pygame
from PIL.Image import frombuffer, fromarray
from numpy import asarray
from pynput import keyboard
from pynput.mouse import Controller
from win32api import RGB
from win32con import SRCCOPY, GWL_EXSTYLE, WS_EX_LAYERED, HWND_TOP, LWA_COLORKEY, SWP_NOSIZE
from win32gui import IsWindowVisible, GetWindowText, GetWindowDC, ReleaseDC, DeleteObject, EnumWindows, GetWindowRect, SetWindowLong, GetWindowLong, SetLayeredWindowAttributes, SetWindowPos
from win32ui import CreateDCFromHandle, CreateBitmap
from datetime import datetime
import torch
import torchvision

DBDhwnd = None
mode = 3  # mode 0 for main run, mode 1 for highlighting node locations on screen, mode 2 for grabbing nodes for pytorch model directory
          # mode 3 for highlighting line boxes on screen, mode 4 for grabbing lines for pytorch model directory
display = 1 # display 1 for show predictions
mouse = Controller()
halt = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# node adj list for ordering nodes
AdjecentDict = {
    0: [17, 6, 7, 8],
    1: [7, 8, 9, 10],
    2: [9, 10, 11, 12],
    3: [11, 12, 13, 14],
    4: [13, 14, 15, 16],
    5: [15, 16, 17, 6],
    6: [18, 19],
    7: [19, 20],
    8: [20, 21],
    9: [21, 22],
    10: [22, 23],
    11: [23, 24],
    12: [24, 25],
    13: [25, 26],
    14: [26, 27],
    15: [27, 28],
    16: [28, 29],
    17: [29, 18]
}

#line coords for node adj list
LineCoordsDict = {
    0:[(563, 458), (584, 424), (635, 425), (657, 456)],
    1:[(693, 474), (736, 476), (759, 524), (743, 557)],
    2:[(743, 600), (755, 633), (735, 677), (694, 682)],
    3:[(662, 705), (635, 733), (584, 738), (559, 704)],
    4:[(528, 682), (483, 680), (457, 633), (478, 600)],
    5:[(477, 559), (463, 522), (485, 477), (529, 476)],
    6:[(506, 338), (583, 317)],
    7:[(638, 313), (718, 337)],
    8:[(771, 363), (828, 424)],
    9:[(860, 474), (875, 548)],
    10:[(884, 604), (859, 687)],
    11:[(830, 733), (772, 793)],
    12:[(721, 820), (639, 843)],
    13:[(583, 845), (504, 818)],
    14:[(449, 795), (388, 733)],
    15:[(364, 683), (340, 605)],
    16:[(348, 549), (381, 477)],
    17:[(390, 424), (451, 365)],
}

# monitor coords of each node spot
webIndex = [(610, 469), (705, 523), (706, 634), (611, 690), (516, 634),
            (516, 523), (555, 367), (666, 368), (771, 424), (828, 524),
            (828, 634), (771, 733), (667, 789), (555, 790), (450, 734),
            (394, 633), (393, 523), (450, 424), (449, 304), (611, 260),
            (773, 304), (892, 421), (935, 578), (891, 736), (774, 853),
            (611, 896), (449, 854), (331, 734), (286, 578), (331, 422)]

# stops the code
def on_press(key):
    global halt
    halt = True


# finds the dbd window handle
def winEnumHandler(hwnd, ctx):
    if IsWindowVisible(hwnd):
        if str(GetWindowText(hwnd)) == 'DeadByDaylight  ':
            global DBDhwnd
            DBDhwnd = hwnd


# grabs the screen
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


# records mouse coords on press and prints at 30
def recordWeb(key):
    global webIndex
    webIndex.append((mouse.position[0], mouse.position[1]))
    if len(webIndex) == 30:
        print(webIndex)

# records mouse coords on press and prints at 48
def recordLine(key):
    global lineIndex
    lineIndex.append((mouse.position[0], mouse.position[1]))
    if len(lineIndex) == 48                                                :
        print(lineIndex)

# transform function for pytorch model
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([.5], [.5])
])

EnumWindows(winEnumHandler, None)
size = GetWindowRect(DBDhwnd)
width = size[2]
height = size[3]

if mode == 0:
    images = []
    nodes = []

    # creates the listner to stop the code
    listener = keyboard.Listener(on_press=on_press)
    print("Press any key to stop")
    listener.start()

    # loads the node model
    NodeModel = torchvision.models.resnet18().to(device)
    NodeModel.load_state_dict(torch.load("NodeModel.pth"))
    NodeModel.eval()

    while True:
        # grabs the screen
        img = grabImage(width, height, 0, 0, DBDhwnd)
        fromarray(img).save("test.png")

        # predicts each node location and outputs
        for x in range(30):
            temp = fromarray(img[webIndex[x][1] - 40:webIndex[x][1] + 40, webIndex[x][0] - 40:webIndex[x][0] + 40])
            output = NodeModel(transform(temp).unsqueeze(0))
            output_idx = torch.argmax(output)
            nodes.append(output_idx)

        #displays the prediction
        if display:
            # pygame init display
            pygame.init()
            screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
            done = False
            fuchsia = (255, 0, 128)  # Transparency color
            blue = (0, 0, 255)
            green = (0, 255, 0)
            white = (255, 255, 255)
            cyan = (0, 255, 255)

            # Create layered window
            hwnd = pygame.display.get_wm_info()["window"]
            SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

            # Set window transparency color
            SetLayeredWindowAttributes(hwnd, RGB(*fuchsia), 0, LWA_COLORKEY)
            SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

            screen.fill(fuchsia)  # Transparent background

            for x in range(30):
                if nodes[x] == 1 or nodes[x] == 0:
                    pygame.draw.arc(screen, blue, pygame.Rect(webIndex[x][0] - 44, webIndex[x][1] - 44, 88, 88), 0, 360, 8)
                elif nodes[x] == 2:
                    pygame.draw.arc(screen, green, pygame.Rect(webIndex[x][0] - 44, webIndex[x][1] - 44, 88, 88), 0, 360, 8)
                else:
                    pygame.draw.arc(screen, white, pygame.Rect(webIndex[x][0] - 44, webIndex[x][1] - 44, 88, 88), 0, 360, 8)

            pygame.display.update()
            sleep(30)
            exit(0)

elif mode == 1:
    # uncomment this line and clear webIndex to record new indexes. Hit any key to store the location of the mouse
    '''with keyboard.Listener(on_press=recordweb) as listener:
            listener.join()'''

    # pygame init display
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

    # draws the nodes in color
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(fuchsia)  # Transparent background

        for x in webIndex:
            pygame.draw.circle(screen, blue, (x[0], x[1]), 36)

        pygame.display.update()

elif mode == 2:
    counter = 0
    time_now = datetime.now()
    current_time = time_now.strftime("%H%M%S")

    # grabs the screen
    img = grabImage(width, height, 0, 0, DBDhwnd)
    fromarray(img).save("test.png")

    # saves the node pics for addition to the machine learning folders
    for x in webIndex:
        temp = fromarray(img[x[1] - 40:x[1] + 40, x[0] - 40:x[0] + 40])
        temp.save('Web/' + str(current_time) + str(counter) + ".png")
        counter += 1

elif mode == 3:
    # uncomment this line and clear LineIndex to record new indexes. Hit any key to store the location of the mouse
    '''with keyboard.Listener(on_press=recordLine) as listener:
            listener.join()'''

    # pygame init display
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    done = False
    fuchsia = (255, 0, 128)  # Transparency color
    blue = (255, 255, 255)

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*fuchsia), 0, LWA_COLORKEY)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    # draws the boxes in color
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(fuchsia)  # Transparent background

        for x in LineCoordsDict:
            for y in LineCoordsDict[x]:
                pygame.draw.rect(screen, blue, pygame.Rect(y[0] - 10, y[1] - 10, 20, 20), 1)

        pygame.display.update()

elif mode == 4:
    counter = 0
    time_now = datetime.now()
    current_time = time_now.strftime("%H%M%S")

    # grabs the screen
    img = grabImage(width, height, 0, 0, DBDhwnd)
    fromarray(img).save("test.png")

    # saves the line pics for addition to the machine learning folders
    for x in LineCoordsDict:
        for y in LineCoordsDict[x]:
            temp = fromarray(img[y[1] - 10:y[1] + 10, y[0] - 10:y[0] + 10])
            temp.save('Web/' + str(current_time) + str(counter) + ".png")
            counter += 1