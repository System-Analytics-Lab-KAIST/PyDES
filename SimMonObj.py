# import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from PyQt5 import uic
import time
import zmq
# import random
import threading
from Graph import *
from Utility import *

#                 0         1        2        3         4       5          6         7           8         9         10          11
# gPalette = [Qt.white,  Qt.black,  Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta, Qt.yellow, Qt.gray, Qt.darkRed, Qt.darkGreen, Qt.darkBlue, Qt.darkCyan, Qt.darkMagenta, Qt.darkYellow, Qt.darkGray]
# Qt_Palette = {s.value: QColor(Colors[s.value]) for s in State}
# Qt_Palette = [QColor(M_Palette[s]) for s in State]
Qt_Palette = [QColor(c) for c in Colors]
gBrushPal = [QBrush(c) for c in Qt_Palette]
gBrushTransparent = QBrush(Qt.transparent)
# gPenPal = [QPen(c) for c in Qt_Palette]
gPenPal = [QPen(c,0) for c in Qt_Palette]  # pen width 0 for constant width
gPenTransparent = QPen(Qt.transparent)



class RootObj(QObject):
    colorSignal = pyqtSignal(list)
    penSignal = pyqtSignal(list)
    moveSignal = pyqtSignal(list)
    animSignal = pyqtSignal(list)
    textSignal = pyqtSignal(list)
    progressSignal = pyqtSignal(list)
    rotateSignal = pyqtSignal(list)
    scaleSignal = pyqtSignal(list)
    removeSignal = pyqtSignal(list)
    queueSignal = pyqtSignal(list)
    enqueueSignal = pyqtSignal(list)
    dequeueSignal = pyqtSignal(list)

    def __init__(self, name, x=0, y=0):
        super().__init__()
        self.name = name
        self.x = x
        self.y = y
        self.org_x = x
        self.org_y = y
        self.shape = None
        self.text = None
        self.colorSignal.connect(self.colorSignalHandler)
        self.penSignal.connect(self.penSignalHandler)
        self.moveSignal.connect(self.moveSignalHandler)
        self.rotateSignal.connect(self.rotateSignalHandler)
        self.scaleSignal.connect(self.scaleSignalHandler)
        self.animSignal.connect(self.animSignalHandler)
        self.textSignal.connect(self.textSignalHandler)
        self.progressSignal.connect(self.progressSignalHandler)
        self.removeSignal.connect(self.removeSignalHandler)
        self.queueSignal.connect(self.queueSignalHandler)
        self.enqueueSignal.connect(self.enqueueSignalHandler)
        self.dequeueSignal.connect(self.dequeueSignalHandler)



    def set_pen_palette(self, pc, shape=None):
        if pc is None:
            pen = gPenTransparent
        else:
            pen =  gPenPal[pc]  # black boundary
        if shape is None:
            shape = self.shape
        shape.setPen(pen)
        if shape == self.shape:
            self.pen = pen

    def set_pen(self, pc, width, shape=None):
        if width == 0:
            self.set_pen_palette(pc, shape)
            return
        if shape is None: shape = self.shape
        if pc < 0:
            color = Qt.transparent
        else:
            color = QColor(Colors[pc])
        pen = QPen(color)
        pen.setWidth(width)
        if shape is None:
            self.shape.setPen(pen)
            self.pen = pen
        else:
            shape.setPen(pen)


    def set_brush(self, bc, shape=None):
        if bc is None or bc < 0:
            brush = gBrushTransparent
        else:
            if bc > Colors_Max: bc = Colors_Max
            brush = gBrushPal[bc]
        if shape is None:
            shape = self.shape
        shape.setBrush(brush)
        if shape == self.shape:
            self.brush = brush


    @pyqtSlot(list)
    def colorSignalHandler(self, m):
        color = int(m[2])
        if len(m) == 3:
            shape = self.shape
        elif m[3] == 'sub':
            shape = self.sub_shape
        else:
            shape = m[3]
        self.set_brush(color, shape)


    @pyqtSlot(list)
    def penSignalHandler(self, m):
        width = int(m[2])
        if len(m) == 3:
            self.set_pen(pc=None, width=width)
        else:
            pc = int(m[3])
            self.set_pen(pc=pc, width=width)

    @pyqtSlot(list)
    def moveSignalHandler(self, m):
        x = float(m[2])
        y = float(m[3])
        self.shape.setPos(x, y)
        self.x = x
        self.y = y
        # self._set_position(x, y)

    @pyqtSlot(list)
    def rotateSignalHandler(self, m):
        self.shape.setRotation(float(m[2]))

    @pyqtSlot(list)
    def scaleSignalHandler(self, m):
        self.shape.setScale(float(m[2]))


    def _set_position(self, position):
        self.shape.setPos(position)

    position = pyqtProperty(QPointF, fset=_set_position)

    @pyqtSlot(list)
    def animSignalHandler(self, m):
        x, y, t = float(m[2]), float(m[3]), float(m[4])
        self.anim = QPropertyAnimation(self, b"position")
        self.anim.setDuration(int(t))
        self.anim.setStartValue(QPointF(self.x, self.y))
        self.anim.setEndValue(QPointF(x, y))
        self.anim.start()
        self.x = x
        self.y = y


    @pyqtSlot(list)
    def textSignalHandler(self, m):
        # if self.text is None: #locals()
        #     self.text = self.addText(txt)
        #     print (f"{self.name} does not have text for {txt}")
        self.text.setPlainText(m[2])

    @pyqtSlot(list)
    def removeSignalHandler(self, m):
        objBook = self.objBook
        objBook.scene.removeItem(self.shape)
        # objBook.dict.pop(self.name)  # TODO monitor crash

    @pyqtSlot(list)
    def progressSignalHandler(self, m):
        self.progressRect(self.shape, self.margin, float(m[2]))


    @pyqtSlot(list)
    def queueSignalHandler(self, m):
        self.setCount(int(m[2]))
        if len(m) > 3:
            self.set_brush(int(m[3]))

    @pyqtSlot(list)
    def enqueueSignalHandler(self, m):
        d = int(m[2]) if len(m)>=3 else 1
        self.setCount(self.qCount+d)

    @pyqtSlot(list)
    def dequeueSignalHandler(self, m):
        d = int(m[2]) if len(m)>=3 else 1
        self.setCount(self.qCount-d)


    def addText(self, txt, size=7, bold=True, w=0, h=0):
        txt = str(txt)
        self.text = QGraphicsTextItem(txt)
        self.text.setParentItem(self.shape)
        x, y = self.x, self.y
        if h > 0: y += (h/2-size)
        if w > 0: x += (w-size*len(txt))/2
        self.text.setPos(x, y)
        f = QFont()
        f.setPointSize(size)
        if bold:
            f.setBold(bold)
        self.text.setFont(f)
        return self.text

    def progressRect(self, shape, margin, u):
        rect = shape.rect()
        if self.wx > self.wy:  # horizontal
            w = (self.wx + margin * 2) * u
            rect.setWidth(w)
        else:
            h = (self.wy + margin * 2) * u
            rect.setHeight(h)
        shape.setRect(rect)

    # Qt application에서 signal handler를 직접 부르면 안되고
    # signal과 handler를 connect해주고,
    # signal을 emit해야 함
    def processMsg(self, m):
        # command = m[0]
        try:
            sig_name = f"{m[0]}Signal"
            sig = getattr(self, sig_name)
            sig.emit(m)
        except KeyError as e:
            print(f"signal for command is not found: {m}")
        return

#################################################
#  Object creation for label text
#     lo = LabelObj(name, x, y, color, size)
#     book.append(lo)
#           (x,y) : start point
#           color : text color
#  Message commands
#     verb name parameters
#        verb : move, anim, color, text
#################################################
class LabelObj(RootObj):
    def __init__(self, name, x=0, y=0, color=1, size=7):
        super().__init__(name, x, y)
        self.text = QGraphicsTextItem(name)
        self.text.setPos(self.x, self.y)
        self.text.setDefaultTextColor(Qt_Palette[color])
        self.shape = self.text
        f = QFont()
        f.setPointSize(size)
        f.setBold(True)
        self.text.setFont(f)

    @pyqtSlot(list)
    def colorSignalHandler(self, m):
        self.text.setDefaultTextColor(Qt_Palette[int(m[2])])


#################################################
#  Object creation for line
#     lo = LineObj(name, x, y, ex, ey, c, showName)
#     book.append(lo)
#           (x,y) : start point
#           ex, ey : end point
#           c : line color
#           showName : show name at start point
#  Message commands
#     verb name parameters
#        verb : move, anim, color, text
#################################################
class LineObj(RootObj):
    def __init__(self, name, x=0, y=0, ex=70, ey=50, c=1, showName=False, head=0):
        super().__init__(name, x, y)
        self.ex = ex
        self.ey = ey
        self.head = head
        self.shape = QGraphicsLineItem(self.x, self.y, self.ex, self.ey)
        self.set_pen_palette(c)
        if showName:
            self.addText(self.name)
        if head > 0:
            self.arrowHead(head)

    def arrowHead(self, d):
        dx, dy = self.ex-self.x, self.ey-self.y
        d2 = d/2
        x1, y1 = rotate(dx, dy, -d, d2, self.ex, self.ey)
        x2, y2 = rotate(dx, dy, -d, -d2, self.ex, self.ey)
        self.ah1 = QGraphicsLineItem(self.ex, self.ey, x1, y1)
        self.ah1.setPen(self.pen)
        self.ah1.setParentItem(self.shape)
        self.ah2 = QGraphicsLineItem(self.ex, self.ey, x2, y2)
        self.ah2.setPen(self.pen)
        self.ah2.setParentItem(self.shape)

    @pyqtSlot(list)
    def colorSignalHandler(self, m):
        color = int(m[2])
        self.set_pen_palette(color)
        self.ah1.setPen(self.pen)
        self.ah2.setPen(self.pen)

    @pyqtSlot(list)
    def textSignalHandler(self, m):
        txt = m[2]
        self.text = self.addText(txt, size=150, bold=False)
        self.text.setPlainText(txt)



#################################################
#  Object creation for box
#     bo = BoxObj(name, x, y, wx, wy, c, showName)
#     book.append(bo)
#           (x,y) : location of queue front
#           wx, wy : Box size
#           c : fill color
#           showName : show name in box
#  Message commands
#     verb name parameters
#        verb : move, anim, color, text
#################################################
class BoxObj(RootObj):
    def __init__(self, name, x=0, y=0, wx=70, wy=50, c=0, showName=False, nameSize=None, margin=0, bc=1):
        super().__init__(name, x, y)
        self.wx = wx
        self.wy = wy
        self.margin = margin
        self.color = c
        self.shape = self.makeShape(x, y, wx, wy)
        self.set_pen_palette(bc)
        self.set_brush(c)
        if showName:
            if nameSize is None:
                nameSize = int(min(wx, wy) / 8)
            self.addText(self.name, nameSize, w=wx, h=wy)

    def makeShape(self, x, y, wx, wy):
        m = self.margin
        return QGraphicsRectItem(x-m, y-m, wx+m*2, wy+m*2)


#################################################
#  Object creation for ellipse
#     eo = EllipseObj(name, x, y, wx, wy, c, showName)
#     book.append(eo)
#           (x,y) : location of queue front
#           wx, wy : Ellipse size
#           c : fill color
#           showName : show name in box
#  Message commands
#     verb name parameters
#        verb : move, anim, color, text
#################################################
class EllipseObj(BoxObj):
    def makeShape(self, x, y, wx, wy):
        m = self.margin
        return QGraphicsEllipseItem(x-m, y-m, wx+m*2, wy+m*2)



#################################################
#  Object creation
#     go = GroupObj(groupName, x, y)
#           (x,y) : location of group
#     go.addToGroup(BoxObj('IO1', 20, 80, 20, 20, c=2))
#     go.addToGroup(BoxObj('IO2', 40, 80, 20, 20, c=3))
#     book.append(go)
#  Message commands
#     move/anim groupName parameters
#     color/text groupName.compName parameters
#################################################
class GroupObj(RootObj):
    def __init__(self, name, x=0, y=0):
        super().__init__(name, x, y)
        self.group ={}
        self.shape = QGraphicsItemGroup()

    def addToGroup(self, obj):
        self.group[obj.name] = obj
        self.shape.addToGroup(obj.shape)

    def findPart(self, partName):
        try:
            p = self.group[partName]
            return p
        except KeyError:  # not found
            print (f"Component object not found : {self.name}.{partName}")
            return None

    def processMsg(self, m):
        # parse index
        m1 = m[1]
        a = m1.find('.')
        if a < 0:
            super().processMsg(m)
        else:
            m1_part = m1[a+1:]
            pobj = self.findPart(m1_part)
            if pobj is not None:  # found
                m[1] = m1_part
                pobj.processMsg(m)


#################################################
#  Object creation
#     bao = ArrayObj(name, x, y, dx, nx, dy, ny, wx, wy, c)
#     book.append(bqo)
#           (x,y) : location of UL corner
#           dx, dy : spacing
#           nx, ny : # of columns & rows
#           wx, wy : box size
#           c : fill color
#           text_size : size of text
#           obj_type : class name of element objects
#           bc : boundary color
#  Message commands
#     verb name[index] parameters
#################################################
class ArrayObj(GroupObj):
    def __init__(self, name, x=0, y=0, dx=0, nx=1, dy=0, ny=1, wx=10, wy=10, c=5, text_size=0, obj_type=BoxObj, bc=1, exc=None, mg=None):
        super().__init__(name, x, y)
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.array = []

        for i in range(ny):
            yi = y + i * dy
            for j in range(nx):
                ij = (i,j)
                if exc and ij in exc:
                    self.array.append(None)
                    continue
                xj = x + j * dx
                k = self.idx(i,j)
                namek = f"{name}{k}"
                bok = obj_type(namek, xj, yi, wx, wy, c, bc=bc)
                if text_size > 0:
                    bok.addText("", text_size)
                self.array.append(bok)
                self.addToGroup(bok)

        if mg:
            box = BoxObj("Boundary", x-mg, y-mg, nx*wx+2*mg, ny*wy+2*mg, c=None)
            self.addToGroup(box)

    def idx(self, i, j):
        return i*self.nx + j

    def processMsg(self, m):
        # parse index
        m1 = m[1]
        a = m1.find('[')+1
        b = m1.find(']')
        if a <= 0:  # array index does not exist
            super().processMsg(m)
            return
        assert b > 0, f"array must be name[i] or name[i,j] {m}"
        c = m1.find(',')
        if c >= 0:  # 2D array
            i = int(m1[a:c])
            j = int(m1[c+1:b])
            k = self.idx(i,j)
        else:
            k = int(m1[a:b])
        bok = self.array[k]
        # bok.stateSignal.emit(int(m[4]))
        bok.processMsg(m)


#################################################
#  Object creation
#     bao = CirArrayObj(name, x, y, dx, nx, dy, ny, wx, wy, ccw, c, text_size, obj_type, bc)
#     book.append(bqo)
#           (x,y) : location of UL corner
#           dx, dy : spacing
#           nx, ny : # of columns & rows
#           wx, wy : box size
#           ccw : ccw flow
#           c : fill color
#           text_size : size of text
#           obj_type : class name of element objects
#           bc : boundary color
#  Message commands
#     verb name[index] parameters
#################################################
class CirArrayObj(ArrayObj):
    def __init__(self, name, x=0, y=0, dx=0, nx=1, dy=0, ny=1, wx=10, wy=10, ccw=True, c=5, text_size=0, obj_type=BoxObj, bc=1):
        super().__init__(name, x, y)
        self.nx = nx
        self.ny = ny
        self.n = 2*(nx + ny - 2)
        self.array = []
        xk, yk = x, y
        for k in range(self.n):
            namek = f"{name}{k}"
            bok = obj_type(namek, xk, yk, wx, wy, c, bc=bc)
            if text_size > 0:
                bok.addText("", text_size)
            self.array.append(bok)
            self.addToGroup(bok)
            xk, yk = circular_position(nx, ny, ccw, dx, dy, xk, yk, k)


    def processMsg(self, m):
        # parse index
        m1 = m[1]
        a = m1.find('[')+1
        b = m1.find(']')
        if a <= 0:  # array index does not exist
            super().processMsg(m)
            return
        assert b > 0, f"array must be name[i] or name[i,j] {m}"
        k = int(m1[a:b])
        bok = self.array[k]
        bok.processMsg(m)

#################################################
#  Object creation
#     go = GridObj(name, x, y, dx, nx, dy, ny, wx, wy, c)
#     book.append(go)
#           (x,y) : location of UL corner
#           dx, dy : spacing
#           nx, ny : # of columns & rows
#           wx, wy : box size
#           c : fill color
#  Message commands : None
#     verb name[index] parameters
#################################################
class GridObj(RootObj):
    def __init__(self, name, x=0, y=0, nx=1, ny=1, wx=10, wy=10, c=5, nameSize=0):
        super().__init__(name, x, y)
        self.shape = QGraphicsItemGroup()
        self.pen =  gPenPal[c]

        dx = wx * nx
        for i in range(ny+1):
            yi = y + i*wy
            line = QGraphicsLineItem(x, yi, x+dx, yi)
            line.setPen(self.pen)
            self.shape.addToGroup(line)

        dy = wy * ny
        for i in range(nx + 1):
            xi = x + i * wx
            line = QGraphicsLineItem(xi, y, xi, y+dy)
            line.setPen(self.pen)
            self.shape.addToGroup(line)

        if nameSize > 0:
            self.addText(self.name, nameSize, w=nx*wx, h=ny*wy)
#################################################
#  Object creation for numeric queue
#     nqo = NumQObj(name, x, y, wx, wy, c)
#     book.append(nqo)
#           (x,y) : location of queue front
#           wx, wy : Ellipse size
#           c : fill color
#  Message commands
#     enqueue name
#     dequeue name
#     queue name len
#################################################


class NumQEllObj(EllipseObj):
    def __init__(self, name, x=0, y=0, wx=70, wy=50, c=0, bc=1):
        nameSize = int(min(wx, wy)/4)
        super().__init__(name, x, y, wx, wy, c, True, nameSize, bc=bc)
        self.setCount(0)
        self.text.setPos(x+wx/4, y+wy/4)

    def setCount(self, qs):
        self.qCount = qs
        self.textSignal.emit([None, None, str(qs)])


class NumQBoxObj(BoxObj):
    def __init__(self, name, x=0, y=0, wx=70, wy=50, c=0, bc=1):
        nameSize =  int(min(wx, wy)/2)
        super().__init__(name, x, y, wx, wy, c, True, nameSize, bc=bc)
        # BoxObj.__init__(self, name, x, y, wx, wy, c, True, nameSize)
        self.setCount(0)
        self.text.setPos(x + wx / 4, y + wy / 4)

    def setCount(self, qs):
        self.qCount = qs
        self.textSignal.emit([None, None, str(qs)])


#################################################
#  Object creation
#     bqo = ArrayQueueObj(name, x, y, w, n, dir, cf)
#     book.append(bqo)
#           (x,y) : location of queue front
#           w : cell size
#           n : # of cells (queue capacity)
#           dir = '+X' : (rear) left-right (front)
#               = '-X' : (rear) right-left (front)
#               = '+Y' : (rear) top-bottom (front)
#               = '-Y' : (rear) bottom-top (front)
#  Message commands
#     enqueue name [color]
#     dequeue name
#################################################
class ArrayQueueObj(ArrayObj):
    def __init__(self, name, x, y, w=10, n=15, dir='+X', cf=5, obj_type=BoxObj):
        wx, wy = w, w
        if dir[1] =='X': # horizontal
            nx = n
            ny = 1
            if dir == '+X' : # Left->Right, front at Right
                wx = -w
                # x += (n-1)*w
                x -= w
        else: # vertical
            nx = 1
            ny = n
            if dir == '+Y' : # Top->Botton, front at Bottom
                wy = -w
                # y += (n - 1) * w
                y -= w
        dx, dy = wx, wy
        super().__init__(name, x, y, dx, nx, dy, ny, w, w, c=0, obj_type=obj_type)
        self.qMax = n
        self.fullColor = cf
        self.colorQ = []

    def emitColor(self, i, c):
        if i < self.qMax:
            ai = self.array[i]
            ai.colorSignal.emit([None, None, c, ai.shape])

    def appendColor(self, c):
        i = len(self.colorQ)
        self.colorQ.append(c)
        self.emitColor(i, c)


    @pyqtSlot(list)
    def queueSignalHandler(self, m):
        ps = len(self.colorQ)
        for i in range(ps):
            self.emitColor(i, 0)  # erase
        self.colorQ = []
        qs = int(m[2])
        for i in range(qs):
            self.appendColor(self.fullColor)

    @pyqtSlot(list)
    def enqueueSignalHandler(self, m):
        if len(m) > 2:
            c = int(m[2])
        else:
            c = self.fullColor
        self.appendColor(c)

    @pyqtSlot(list)
    def dequeueSignalHandler(self, m):
        if len(self.colorQ) == 0:
            return
        self.colorQ.pop(0)
        n = min(len(self.colorQ), self.qMax)
        for i in range(n):
            self.emitColor(i, self.colorQ[i])
        self.emitColor(n, 0)  # erase

class MonoQueueObj(ArrayQueueObj):
    @pyqtSlot(list)
    def queueSignalHandler(self, m):
        ps = len(self.colorQ)
        qs = int(m[2])
        if ps > qs: # decrease
            for i in range(qs, ps):
                self.emitColor(i, 0)  # erase
                self.colorQ.pop()
        else: # increase
            for i in range(ps, qs):
                self.appendColor(self.fullColor)

    @pyqtSlot(list)
    def enqueueSignalHandler(self, m):
        c = self.fullColor
        self.appendColor(c)

    @pyqtSlot(list)
    def dequeueSignalHandler(self, m):
        if len(self.colorQ) == 0:
            return
        self.colorQ.pop(0)
        n = min(len(self.colorQ), self.qMax)
        self.emitColor(n, 0)  # erase


#################################################
#  Object creation
#     g = GraphObj('Network')
#     g.readGraph('vertex.txt', 'edge.txt')
#     book.append(g)
#  Message commands
#
#################################################
class GraphObj(GroupObj):
    def readGraph(self, vFile, eFile, split=0, scale=1):
        g = Graph(False)
        self.graph = g
        g.readGraph(vFile, eFile, split=split, scale=scale)
        self.addVandE()

    def addVandE(self):
        g = self.graph
        ah = 200
        s = ah*2
        # s = ah
        s2 = s*2
        # for v in g.Vmap.values():
        for v in g.Vmap.values():
            v_is_port = v.is_port()
            mark_vtx = v_is_port
            # mark_vtx = (v.type != 'I')
            # mark_vtx = True
            if mark_vtx:
                vobj = EllipseObj(str(v), v.x - s, v.y - s, s2, s2, c=0, showName=True, nameSize=ah)
                self.addToGroup(vobj)
        for e in g.Elist:
            if e.sv.type is None or e.ev.type is None:
                continue
            c = 3 if e.pair else 1
            eobj = LineObj(f"{e.id()}", e.sv.x, e.sv.y, e.ev.x, e.ev.y, c=c, head=ah)
            # print (e.sv.name, e.ev.name)
            self.addToGroup(eobj)


# import locked_dict.locked_dict as locked_dict
class ObjBook:
    def __init__(self, mainWnd, view, scene):
        self.dict = {}
        self.recyclage = []
        self.n_recycled = 0
        self.recyclage_obj_name = ""
        self.recyclage_x = 0
        self.recyclage_y = 0
        self.mainWnd = mainWnd
        self.view = view
        self.scene = scene


    def append(self, obj):
        self.dict[obj.name] = obj
        self.scene.addItem(obj.shape)
        # self.scene.update()


    # def remove(self, obj):
    #     try:
    #         self.dict.pop(obj.name)
    #         obj.objBook = self
    #         obj.removeSignal.emit([])
    #     except KeyError:
    #         pass
    #     # return # TODO : monitor crash check

    def widgetMsg(self, m):
        if m[0] != 'widget':  return False
        # print(m)
        self.mainWnd.setWidgetSignal.emit(m)
        return True

    # def make_obj_on_demand(self, name):
    #     name4 = name[:4]
    #     if name4 == "Cart":
    #         c = CartObj.colorMap[name[5]]
    #         mo = CartObj(name, 0, 0, wx=10, wy=10, c=c, bc=9)
    #     elif name4 == "Tyre":
    #         mo = TyreObj(name)
    #     else:
    #         return None
    #     self.append(mo)
    #     print (f"Object {name} is created")
    #     return mo

    def get_object_from_recyclage(self, name):
        if name[:4] != self.recyclage_obj_name:
            return None
        if len(self.recyclage) == 0:
            print (f"Recylage for {name} is empty.  Make more..")
            return None
        obj = self.recyclage.pop()
        obj.set_name(name)
        self.append(obj)
        # print(f"{obj.name} is obtained. len = {len(self.recyclage)}, recycled = {self.n_recycled}")
        return obj

    def recycle_object(self, obj, m):
        if obj.name[:4] != self.recyclage_obj_name:
            return False
        try:
            obj.moveSignal.emit([None, None, self.recyclage_x, self.recyclage_y])  # move to recyclage
            self.dict.pop(obj.name)
        except KeyError as e:
            print (f"KeyError {e}")
            return False
        self.recyclage.append(obj)
        self.n_recycled += 1
        # print(f"{obj.name} is recycled. len = {len(self.recyclage)}, recycled = {self.n_recycled}")
        return True

    def get_object(self, name):
        try:
            return self.dict[name]
        except KeyError:  # object not defined
            # TODO monitor crash check
            # obj = self.make_obj_on_demand(name)
            obj = self.get_object_from_recyclage(name)
            if obj:
                return obj

            for k, c in enumerate(name):
                if c == '.' or c == '[':  # group or array
                    return self.get_object(name[:k])
            return None

    def processMsg(self, msg):
        # print(msg)
        self.msg = msg
        m = msg.split()
        if self.widgetMsg(m): return
        obj = self.get_object(m[1])
        # if isinstance(obj, CartObj) and m[0] == 'progress':
        #     return
        if obj is None:
            print("Object not found: ", msg)
        else:
            if m[0] == "remove":
                self.recycle_object(obj, m)
            else:
                obj.processMsg(m)
            self.scene.update()
        # time.sleep(0.001) # remove this line to improve the monitor speed ###


class LogPlayer(threading.Thread):
    def __init__(self, mainWnd, logFile, timeScale):
        super().__init__()
        self.mainWnd = mainWnd
        self.book = mainWnd.book
        self.logFileName = logFile
        self.timeScale = timeScale
        self.Tsince = -1

    def changeTimeScale(self, timeScale):
        self.timeScale = timeScale
        self.Tsince = self.Tnow
        self.timeSince = time.time()

    def run(self):
        self.mainWnd.logThreadRunning = True
        f = open(self.logFileName, 'r')

        self.timeSince = time.time()
        for line in f:
            # print(line)
            self.Tnow = float(line[:12])
            if self.timeScale > 0:
                if self.Tsince < 0:
                    self.Tsince = self.Tnow
                else:
                    dt = (self.Tnow - self.Tsince) / self.timeScale - (time.time() - self.timeSince)
                    if dt > 0:
                        if dt > 1: dt = 1
                        time.sleep(dt)
            self.book.processMsg(line[13:])
            if self.mainWnd.paused:
                self.mainWnd.pauseLock.acquire()
                self.changeTimeScale(self.timeScale)
                self.mainWnd.pauseLock.release()
            if not self.mainWnd.logThreadRunning:
                break
        f.close()
