import os
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
mode = 0 #mode 0 for predicting nodes, mode 1 for highlighting node locations on screen, mode 2 for grabbing nodes for pytorch model directory
mouse = Controller()
halt = False

#node adj list for ordering nodes
AdjecentDict = {
    1: [18, 7, 8, 9],
    2: [8, 9, 10, 11],
    3: [10, 11, 12, 13],
    4: [12, 13, 14, 15],
    5: [14, 15, 16, 17],
    6: [16, 17, 18, 7],
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
    17: [29, 30],
    18: [30, 19]
}

#monitor coords of each node spot
webIndex = [(610, 469), (705, 523), (706, 634), (611, 690), (516, 634), (516, 523), (555, 367), (666, 368), (771, 424),
            (828, 524),
            (828, 634), (771, 733), (667, 789), (555, 790), (450, 734), (394, 633), (393, 523), (450, 424), (449, 304),
            (611, 260),
            (773, 304), (892, 421), (935, 578), (891, 736), (774, 853), (611, 896), (449, 854), (331, 734), (286, 578),
            (331, 422)]

#stops the code
def on_press(key):
    global halt
    halt = True

#finds the dbd window handle
def winEnumHandler(hwnd, ctx):
    if IsWindowVisible(hwnd):
        if str(GetWindowText(hwnd)) == 'DeadByDaylight  ':
            global DBDhwnd
            DBDhwnd = hwnd

#grabs the screen
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

#records mouse coords on press and prints at 30
def recordWeb(key):
    global webIndex
    webIndex.append((mouse.position[0], mouse.position[1]))
    if len(webIndex) == 30:
        print(webIndex)

#transform function for pytorch model
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([.5], [.5])
])

EnumWindows(winEnumHandler, None)
size = GetWindowRect(DBDhwnd)
width = size[2]
height = size[3]

if mode == 0:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    counter = 0
    images = []
    nodes = []

    #creates the listner to stop the code
    listener = keyboard.Listener(on_press=on_press)
    print("Press any key to stop")
    listener.start()

    #loads the model
    model = torchvision.models.resnet18().to(device)
    model.load_state_dict(torch.load("Model.pth"))
    model.eval()

    #grabs the screen
    img = grabImage(width, height, 0, 0, DBDhwnd)
    fromarray(img).save("test.png")

    #clears web directory
    for f in os.listdir('web'):
        os.remove(os.path.join('web', f))

    #predicts each node location and outputs
    for x in range(30):
        temp = fromarray(img[webIndex[x][1] - 40:webIndex[x][1] + 40, webIndex[x][0] - 40:webIndex[x][0] + 40])
        temp.save("web/" + str(x) + ".png")
        output = model(transform(temp).unsqueeze(0))
        output_idx = torch.argmax(output)
        nodes.append(output_idx)

    # pygame init display
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    done = False
    fuchsia = (255, 0, 128)  # Transparency color
    blue = (0, 0, 255)
    green = (0, 255, 0)

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

        for x in range(30):
            if nodes[x] == 0:
                pygame.draw.arc(screen, blue, pygame.Rect(webIndex[x][0] - 40, webIndex[x][1] - 40, 80, 80) , 0, 360, 5)
            else:
                pygame.draw.arc(screen, green, pygame.Rect(webIndex[x][0] - 40, webIndex[x][1] - 40, 80, 80) , 0, 360, 5)
        pygame.display.update()

elif mode == 1:
    # uncomment this line and clear webIndex to record new indexes. Hit any key to store the location of the mouse
    '''with keyboard.Listener(on_press=recordweb) as listener:
            listener.join()'''

    #pygame init display
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

    #draws the nodes in color
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

    #grabs the screen
    img = grabImage(width, height, 0, 0, DBDhwnd)
    fromarray(img).save("test.png")

    #saves the node pics for addition to the machine learning folders
    for x in webIndex:
        temp = fromarray(img[x[1] - 40:x[1] + 40, x[0] - 40:x[0] + 40])
        temp.save('web/' + str(current_time) +str(counter) + ".png")
        counter += 1