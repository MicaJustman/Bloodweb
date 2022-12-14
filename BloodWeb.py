from operator import itemgetter
from time import sleep
import imagehash
import pygame
from PIL.Image import frombuffer, fromarray, open as PILopen
from numpy import asarray
from pynput import keyboard
from pynput.mouse import Controller
from win32api import RGB
from win32con import SRCCOPY, GWL_EXSTYLE, WS_EX_LAYERED, HWND_TOP, LWA_COLORKEY, SWP_NOSIZE, LWA_ALPHA
from win32gui import IsWindowVisible, GetWindowText, GetWindowDC, ReleaseDC, DeleteObject, EnumWindows, GetWindowRect, SetWindowLong, GetWindowLong, SetLayeredWindowAttributes, SetWindowPos
from win32ui import CreateDCFromHandle, CreateBitmap
from datetime import datetime
import torch
import torchvision

DBDhwnd = None
mode = 0  # mode 0 for main run, mode 1 for highlighting node locations on screen, mode 2 for grabbing nodes for pytorch model directory
          # mode 3 for highlighting line boxes on screen, mode 4 for grabbing lines for pytorch model directory
mouse = Controller()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def __str__(self):
        return "value % s ---- children %s" % (self.number, self.children)

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

#hashes of every node connection
LineHashes = {'0-17': ['1f03f01ec0f8deee', '01c0f83f03000000'], '0-6': ['279393c9cde4f6f2', '303019180c048683'],
              '0-7': ['e6e4c9d993b3676f', '060c0c1818307060'], '0-8': ['fef0833ef8811e79', '0000033ff8c00000'],
              '1-7': ['b39393d3dbcbcdcd', '3030181818180c0c'], '1-8': ['f2a6ecc99bb3676f', '07060c1c18307060'],
              '1-9': ['ffff00ff6000ffff', '000000ffff000000'], '1-10': ['0fc77038cce673b9', '80c070381c0e0301'],
              '2-9': ['f9f3e6cc1870e480', '0103061c3870e0c0'], '2-10': ['80f900e0ff0000ff', '000000dfff000000'],
              '2-11': ['c861203098c8cc66', 'c060603018180c06'], '2-12': ['0c1919d898901030', '0c18181818303030'],
              '3-11': ['3a83f07f03e0b8cc', '0080f07f0f000000'], '3-12': ['6535b098c9ece4f6', '60303018180c0e06'],
              '3-13': ['e6cccd999333276f', '060c0c1818307060'], '3-14': ['fff0031ff8811f7f', '0000073ff8c00000'],
              '4-13': ['b393dad9c9c9cded', '3030181818080c0c'], '4-14': ['f9f3f2e4ccc89131', '010306060c181830'],
              '4-15': ['ffff0000ff0000ff', '000000e0ff000000'], '4-16': ['9fc662319cc6e3f9', '80e070381c0e0301'],
              '5-15': ['f3e7ce8c3160c89e', '03070e1c3060c080'], '5-16': ['fffb0000ff0000ff', '00000000ff000000'],
              '5-17': ['6727939bc9ece6f2', '603030180c0c0603'], '5-6': ['c9dbdb9393b7b727', '0818181810303030'],
              '6-18': ['3f9fc7311cc6e3f8', '0180e0701c0f0300'], '6-19': ['e4ccc993b327676f', '0c0c181830306060'],
              '7-19': ['6727b393d9c9c4e6', '60303018180c0c06'], '7-20': ['fcf1e78c38e3cf1f', '0001071e38e0c000'],
              '8-20': ['c9c9c9c9c9c9c9c9', '1818181818181818'], '8-21': ['ff0000ff0000ffff', '000000ffff000000'],
              '9-21': ['e6cc091b33266fcf', '060c1818307060c0'], '9-22': ['3e86e3781e070100', '0080e0781e070100'],
              '10-22': ['fef8e38f38e18700', '0001030f3cf0c000'], '10-23': ['303018180c060603', '7030181c0c060603'],
              '11-23': ['00000000ff81003c', '00000000ffff0000'], '11-24': ['1819db181a1a1818', '1818181818181818'],
              '12-24': ['0000c0e0388e4701', '0000c0f0381e0701'], '12-25': ['e6ecccd893302060', '060c0c1818303060'],
              '13-25': ['20b190d8c86c6636', '303018180c0c0606'], '13-26': ['fe98d1c71c30e080', '000003071c78e0c0'],
              '14-26': ['dbdbdbdbdbdb9bdb', '1818181818181818'], '14-27': ['ffdf0000ff0000ff', '00000000fffe0000'],
              '15-27': ['e6eccdd99333674f', '060c0c18303060e0'], '15-28': ['7f1fc7711c87e1f8', '0100c0f03c0f0300'],
              '16-28': ['f8e38f38e1871f7f', '00030f3cf0c00000'], '16-29': ['4f6727b39bc9cce4', 'c0603030181c0c06'],
              '17-29': ['ff0000ff0000ffff', '000000ffff000000'], '17-18': ['9393939393939393', '3030101010101010']}


# monitor coords of each node spot
webIndex = [(610, 469), (705, 523), (706, 634), (611, 690), (516, 634),
            (516, 523), (555, 367), (666, 368), (771, 424), (828, 524),
            (828, 634), (771, 733), (667, 789), (555, 790), (450, 734),
            (394, 633), (393, 523), (450, 424), (449, 304), (611, 260),
            (773, 304), (892, 421), (935, 578), (891, 736), (774, 853),
            (611, 896), (449, 854), (331, 734), (286, 578), (331, 422)]

# stops the code
def on_press(key):
    exit(0)


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

if mode == 0:
    # creates the listner to stop the code
    listener = keyboard.Listener(on_press=on_press)
    print("Press any key to stop")
    listener.start()

    # loads the node model
    NodeModel = torchvision.models.resnet18().to(device)
    NodeModel.load_state_dict(torch.load("NodeModel.pth"))
    NodeModel.eval()

    while True:
        nodes = []
        lines = {}
        base = [0, 1, 2, 3, 4, 5]
        baseOpt = []
        order = []
        trees = []
        visited = []

        # grabs the screen
        img = grabImage(width, height, 0, 0, DBDhwnd)
        fromarray(img).save("test.png")

        # predicts each node location and outputs
        for x in range(30):
            temp = fromarray(img[webIndex[x][1] - 40:webIndex[x][1] + 40, webIndex[x][0] - 40:webIndex[x][0] + 40])
            output = NodeModel(transform(temp).unsqueeze(0))
            output_idx = torch.argmax(output)
            nodes.append(output_idx)

        #id the graph lines
        for x in AdjecentDict:
            for y in AdjecentDict[x]:
                coord = LineCoordsDict[str(x) + '-' + str(y)]
                Gray = imagehash.hex_to_hash(LineHashes[str(x) + '-' + str(y)][0])
                Gold = imagehash.hex_to_hash(LineHashes[str(x) + '-' + str(y)][1])
                temp = imagehash.average_hash(fromarray(img[coord[1] - 10:coord[1] + 10, coord[0] - 10:coord[0] + 10]))

                if abs(temp - Gray) < 10 or abs(temp - Gold) < 10:
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
                imgHash = str(imagehash.average_hash(fromarray(img[webIndex[x][1] - 40:webIndex[x][1] + 40, webIndex[x][0] - 40:webIndex[x][0] + 40])))
                rootNode = treeNode(x, imagehash)

                for y in AdjecentDict[x]:
                    if (nodes[y] == 2 or nodes[y] == 3) and (lines[str(x) + '-' + str(y)] == 1) and y not in visited:
                        imgHash = str(imagehash.average_hash(fromarray(img[webIndex[y][1] - 40:webIndex[y][1] + 40, webIndex[y][0] - 40:webIndex[y][0] + 40])))
                        midNode = treeNode(y, imgHash)
                        rootNode.addNode(midNode)
                        visited.append(y)

                        for z in AdjecentDict[y]:
                            if (nodes[z] == 2 or nodes[z] == 3) and (lines[str(y) + '-' + str(z)] == 1) and z not in visited:
                                imgHash = str(imagehash.average_hash(fromarray(img[webIndex[z][1] - 40:webIndex[z][1] + 40, webIndex[z][0] - 40:webIndex[z][0] + 40])))
                                leafNode = treeNode(z, imgHash)
                                midNode.addNode(leafNode)
                                visited.append(z)

                trees.append(rootNode)

        for x in trees:
            print(x.listNodes())

        #Displays the nodes and graph lines
        #displayBasic()

        #Displays the trees
        displayTrees()

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
    fromarray(img).save("test.png")

    # saves the line pics for addition to the machine learning folders
    '''for x in AdjecentDict:
        for y in AdjecentDict[x]:
            coord = LineCoordsDict[str(x) + '-' + str(y)]
            temp = fromarray(img[coord[1] - 10:coord[1] + 10, coord[0] - 10:coord[0] + 10])
            temp.save('Web/' + str(current_time) + str(counter) + ".png")
            counter += 1'''

    #prints image hash for the dict
    for x in AdjecentDict:
        for y in AdjecentDict[x]:
            hashes[str(x) + '-' + str(y)] = [str(imagehash.average_hash(PILopen('Line/GrayLines/' + str(counter) + '.png'))), str(imagehash.average_hash(PILopen('Line/GoldLines/' + str(counter) + '.png')))]
            counter += 1
    print(hashes)
