import sys
from operator import itemgetter
from time import sleep
import imagehash
import pygame
from PIL.Image import frombuffer, fromarray, open as PILopen
from numpy import asarray
from pynput import keyboard
from pynput.mouse import Controller, Button
from win32api import RGB
from win32con import SRCCOPY, GWL_EXSTYLE, WS_EX_LAYERED, HWND_TOP, LWA_COLORKEY, SWP_NOSIZE, LWA_ALPHA
from win32gui import IsWindowVisible, GetWindowText, GetWindowDC, ReleaseDC, DeleteObject, EnumWindows, GetWindowRect, SetWindowLong, GetWindowLong, SetLayeredWindowAttributes, SetWindowPos
from win32ui import CreateDCFromHandle, CreateBitmap
from datetime import datetime
import torch
import torchvision
from SiameseClassify import SiameseNetwork
import torch.nn.functional as F

character = "Executioner"
mode = 0  # mode 0 for main run, mode 1 for highlighting node locations on screen, mode 2 for grabbing nodes for pytorch model directory
          # mode 3 for highlighting line boxes on screen, mode 4 for grabbing lines for pytorch model directory

DBDhwnd = None
mouse = Controller()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
halt = False

class treeNode:
    def __init__(self, number, hash):
        self.number = number
        self.hash = hash
        self.children = []
        self.parent = None

    def addNode(self, node):
        node.parent = self
        self.children.append(node)

    def listNodes(self):
        nodes = [self.number]

        for x in self.children:
            nodes.append(x.number)

            for y in x.children:
                nodes.append(y.number)

        return nodes

    def search(self, hash):
        path = []
        for y in hash:
            stack = [self]

            while len(stack) != 0:
                current = stack.pop(0)

                for x in current.children:
                    stack.append(x)

                dif = abs(imagehash.hex_to_hash(current.hash) - imagehash.hex_to_hash(y))
                if dif < 25:
                    if current.number not in path:
                        path.append(current.number)
                    if current.parent is not None:
                        if current.parent.number not in path:
                            path.append(current.parent.number)
                        if current.parent.parent is not None:
                            if current.parent.parent.number not in path:
                                path.append(current.parent.parent.number)

        return path

    def __str__(self):
        return "value % s --- hash %s --- children %s" % (self.number, self.hash, self.children)

    def __repr__(self):
        return str(self)

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
    '0-17': (563, 458),
    '0-6': (584, 424),
    '0-7': (635, 425),
    '0-8': (657, 456),
    '1-7': (693, 474),
    '1-8': (736, 476),
    '1-9': (759, 524),
    '1-10': (743, 557),
    '2-9': (743, 600),
    '2-10': (755, 633),
    '2-11': (735, 677),
    '2-12': (694, 682),
    '3-11': (662, 705),
    '3-12': (635, 733),
    '3-13': (584, 738),
    '3-14': (559, 704),
    '4-13': (528, 682),
    '4-14': (483, 680),
    '4-15': (457, 633),
    '4-16': (478, 600),
    '5-15': (477, 559),
    '5-16': (463, 522),
    '5-17': (485, 477),
    '5-6': (529, 476),
    '6-18': (506, 338),
    '6-19': (583, 317),
    '7-19': (638, 313),
    '7-20': (718, 337),
    '8-20': (771, 363),
    '8-21': (828, 424),
    '9-21': (860, 474),
    '9-22': (875, 548),
    '10-22': (884, 604),
    '10-23': (859, 687),
    '11-23': (830, 733),
    '11-24': (772, 793),
    '12-24': (721, 820),
    '12-25': (639, 843),
    '13-25': (583, 845),
    '13-26': (504, 818),
    '14-26': (449, 795),
    '14-27': (388, 733),
    '15-27': (364, 683),
    '15-28': (340, 605),
    '16-28': (348, 549),
    '16-29': (366, 477),
    '17-29': (390, 424),
    '17-18': (451, 365),
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

# transform function for node model
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([.5], [.5])
])

# transform function for Line model
transformLine = torchvision.transforms.Compose([
    torchvision.transforms.Resize((100, 100)),
    torchvision.transforms.ToTensor()
])

EnumWindows(winEnumHandler, None)
size = GetWindowRect(DBDhwnd)
width = size[2]
height = size[3]

# displays the prediction
def displayBasic():
    # pygame init display
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    Black = (0, 0, 0)  # Transparency color
    blue = (0, 0, 255)
    green = (0, 255, 0)
    white = (255, 255, 255)

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*Black), 100, LWA_ALPHA)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    screen.fill(Black)  # Transparent background

    for x in range(30):
        if nodes[x] == 1 or nodes[x] == 0:
            pygame.draw.arc(screen, blue, pygame.Rect(webIndex[x][0] - 44, webIndex[x][1] - 44, 88, 88), 0, 360, 8)
        elif nodes[x] == 2:
            pygame.draw.arc(screen, green, pygame.Rect(webIndex[x][0] - 44, webIndex[x][1] - 44, 88, 88), 0, 360, 8)
        else:
            pygame.draw.arc(screen, white, pygame.Rect(webIndex[x][0] - 44, webIndex[x][1] - 44, 88, 88), 0, 360, 8)

    for x in AdjecentDict:
        for y in AdjecentDict[x]:
            coord = LineCoordsDict[str(x) + '-' + str(y)]
            if lines[str(x) + '-' + str(y)] == 1:
                pygame.draw.circle(screen, green, (coord[0], coord[1]), 5)
            else:
                pygame.draw.circle(screen, blue, (coord[0], coord[1]), 5)

    pygame.display.update()
    sleep(360)
    exit(0)

def displayTrees():
    # pygame init display
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    Black = (0, 0, 0)  # Transparency color
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (255, 255, 0), (255, 100, 0), (255, 255, 255)]

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*Black), 100, LWA_ALPHA)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    screen.fill(Black)  # Transparent background

    count = 0
    for x in trees:
        for b in x.children:
            pygame.draw.line(screen, colors[count], webIndex[x.number], webIndex[b.number], width = 6)
            for c in b.children:
                pygame.draw.line(screen, colors[count], webIndex[b.number], webIndex[c.number], width = 6)

        for y in x.listNodes():
            pygame.draw.arc(screen, colors[count], pygame.Rect(webIndex[y][0] - 44, webIndex[y][1] - 44, 88, 88), 0, 360, 8)
        count += 1

    pygame.display.update()
    sleep(360)
    exit(0)

#mouse controll
def mouseControl(x, y):
    temp = mouse.position
    dif = (x - temp[0], y - temp[1])
    mouse.move(dif[0], dif[1])
    sleep(.4)
    mouse.press(Button.left)
    sleep(.6)
    mouse.release(Button.left)
    sleep(.2)

if mode == 0:
    hashes = []

    # creates the listner to stop the code
    listener = keyboard.Listener(on_press=on_press)
    print("Press any key to stop")
    listener.start()

    # loads the node model
    NodeModel = torchvision.models.resnet18().to(device)
    NodeModel.load_state_dict(torch.load("NodeModel.pth"))
    NodeModel.eval()

    # loads the line model
    LineModel = SiameseNetwork().to(device)
    LineModel.load_state_dict(torch.load("LineModel.pth"))
    LineModel.eval()

    #loads priority hashes
    for x in range(1, 10):
        try:
            hashes.append(str(imagehash.average_hash(PILopen('Priority/' + str(x) + '.png'), hash_size=16)))
        except:
            pass

    for x in range(1, 10):
        try:
            hashes.append(str(imagehash.average_hash(PILopen('Priority/' + character + '/' + str(x) + '.png'), hash_size=16)))
        except:
            pass

    while True:
        nodes = []
        lines = {}
        base = [0, 1, 2, 3, 4, 5]
        baseOpt = []
        order = []
        trees = []
        visited = []
        finalOrder = []

        # grabs the screen
        img = grabImage(width, height, 0, 0, DBDhwnd)

        # predicts each node location and outputs
        for x in range(30):
            temp = fromarray(img[webIndex[x][1] - 40:webIndex[x][1] + 40, webIndex[x][0] - 40:webIndex[x][0] + 40])
            output = NodeModel(transform(temp).unsqueeze(0))
            output_idx = torch.argmax(output)
            nodes.append(output_idx)

        #id the graph lines
        saved = transformLine(PILopen('Line/Lines/0.png').convert("L")).to(device)
        for x in AdjecentDict:
            for y in AdjecentDict[x]:
                coord = LineCoordsDict[str(x) + '-' + str(y)]
                temp = transformLine(fromarray(img[coord[1] - 10:coord[1] + 10, coord[0] - 10:coord[0] + 10]).convert("L")).to(device)

                output1, output2 = LineModel(saved.unsqueeze(0), temp.unsqueeze(0))
                euclidean_distance = F.pairwise_distance(output1, output2)

                if euclidean_distance < .5:
                    lines[str(x) + '-' + str(y)] = 1
                else:
                    lines[str(x) + '-' + str(y)] = 0

        #sort tree roots based on connection number
        for x in base:
            count = 0

            for y in AdjecentDict[x]:
                if (nodes[y] == 2 or nodes[y] == 3) and (lines[str(x) + '-' + str(y)] == 1):
                    count += 1

            order.append([x, count])

        order.sort(key=itemgetter(1), reverse=True)

        for x in order:
            baseOpt.append(x[0])

        #creates the trees
        for x in baseOpt:
            if nodes[x] == 2 or nodes[x] == 3:
                imgHash = str(imagehash.average_hash(fromarray(img[webIndex[x][1] - 40:webIndex[x][1] + 40, webIndex[x][0] - 40:webIndex[x][0] + 40]), hash_size=16))
                rootNode = treeNode(x, imgHash)

                for y in AdjecentDict[x]:
                    if (nodes[y] == 2 or nodes[y] == 3) and (lines[str(x) + '-' + str(y)] == 1) and y not in visited:
                        imgHash = str(imagehash.average_hash(fromarray(img[webIndex[y][1] - 40:webIndex[y][1] + 40, webIndex[y][0] - 40:webIndex[y][0] + 40]), hash_size=16))
                        midNode = treeNode(y, imgHash)
                        rootNode.addNode(midNode)
                        visited.append(y)

                        for z in AdjecentDict[y]:
                            if (nodes[z] == 2 or nodes[z] == 3) and (lines[str(y) + '-' + str(z)] == 1) and z not in visited:
                                imgHash = str(imagehash.average_hash(fromarray(img[webIndex[z][1] - 40:webIndex[z][1] + 40, webIndex[z][0] - 40:webIndex[z][0] + 40]), hash_size=16))
                                leafNode = treeNode(z, imgHash)
                                midNode.addNode(leafNode)
                                visited.append(z)

                trees.append(rootNode)
        print(trees)

        '''for x in trees:
            print(x.listNodes())
            for y in x.listNodes():
                finalOrder.append(y)

        for x in finalOrder:
                if halt:
                    exit(1)
                if nodes[x] == 2:
                    mouseControl(webIndex[x][0], webIndex[x][1])

        sleep(2)
        mouse.position = (610, 570)
        sleep(.1)
        mouse.press(Button.left)
        sleep(8)
        mouse.release(Button.left)'''


        #Displays the nodes and graph lines
        displayBasic()

        #Displays the trees
        #displayTrees()
        exit(1)

elif mode == 1:
    # uncomment this line and clear webIndex to record new indexes. Hit any key to store the location of the mouse
    '''with keyboard.Listener(on_press=recordweb) as listener:
            listener.join()'''

    # pygame init display
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME)
    done = False
    Black = (0, 0, 0)  # Transparency color
    blue = (0, 0, 255)

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*Black), 100, LWA_ALPHA)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    # draws the nodes in color
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(Black)  # Transparent background

        for x in webIndex:
            pygame.draw.circle(screen, blue, (x[0], x[1]), 36)

        pygame.display.update()

elif mode == 2:
    counter = 0
    time_now = datetime.now()
    current_time = time_now.strftime("%H%M%S")

    # grabs the screen
    img = grabImage(width, height, 0, 0, DBDhwnd)

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
    Black = (0, 0, 0)  # Transparency color
    blue = (255, 255, 255)

    # Create layered window
    hwnd = pygame.display.get_wm_info()["window"]
    SetWindowLong(hwnd, GWL_EXSTYLE, GetWindowLong(hwnd, GWL_EXSTYLE) | WS_EX_LAYERED)

    # Set window transparency color
    SetLayeredWindowAttributes(hwnd, RGB(*Black), 100, LWA_ALPHA)
    SetWindowPos(hwnd, HWND_TOP, 0, 0, width, height, SWP_NOSIZE)

    # draws the boxes in color
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(Black)  # Transparent background

        for x in AdjecentDict:
            for y in AdjecentDict[x]:
                coord = LineCoordsDict[str(x) + '-' + str(y)]
                pygame.draw.rect(screen, blue, pygame.Rect(coord[0] - 10, coord[1] - 10, 20, 20), 1)

        pygame.display.update()

elif mode == 4:
    counter = 0
    hashes = {}

    time_now = datetime.now()
    current_time = time_now.strftime("%H%M%S")

    # grabs the screen
    img = grabImage(width, height, 0, 0, DBDhwnd)

    # saves the line pics for addition to the machine learning folders
    for x in AdjecentDict:
        for y in AdjecentDict[x]:
            coord = LineCoordsDict[str(x) + '-' + str(y)]
            temp = fromarray(img[coord[1] - 10:coord[1] + 10, coord[0] - 10:coord[0] + 10])
            temp.save('Web/' + str(counter) + ".png")
            counter += 1

