import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic
from SimMonObj import ObjBook, LogPlayer

from ConnectMQ import *
from time import *

class SimAnimWindow(QMainWindow):
    setWidgetSignal = pyqtSignal(list)

    def __init__(self, uic_name):
        super().__init__()
        # UI Layout should be loaded here
        if uic_name is None:
            uic_name = "PyDES/DefaultMon.ui"
        uic.loadUi(uic_name, self)
        # uic.loadUi('HantaShopMon.ui', self)
        # self.centralWidget : windowd의 main에 있는 widget
        # self.bottomWidgets : 아레쪽에 button들 모아 놓은 widget
        # self.rightWidgets : 오른쪽에 button들 모아 놓은 widget
        self.viewport.initViewport2D(self)

        self.msgThreadRunning = False
        self.logThreadRunning = False
        self.paused = False
        self.dropping = False
        # self.pauseLock = threading.Condition(threading.Lock())
        self.pauseLock = threading.Lock()
        self.pauseButton.clicked.connect(self.pauseButtonClicked)
        self.connectButton.clicked.connect(self.connectButtonClicked)
        self.dropButton.clicked.connect(self.dropButtonClicked)
        self.playLogButton.clicked.connect(self.playLogButtonClicked)
        self.speedSlider.valueChanged.connect(self.speedSliderChanged)
        self.setWidgetSignal.connect(self.setWidgetHandler)

        self.monMQ = MonitorMQ(self)
        self.connectURL.setText(self.monMQ.url)

    def getWidget(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None


    def getTimeScale(self):
        v = self.speedSlider.value()
        # f = v / 10
        f = v
        if v == 1000:
            f = 0
        return f

    def speedSliderChanged(self):
        f = self.getTimeScale()
        if f == 0:
            s = 'Speed factor (AFAP)'
        else:
            s = f'Speed factor({f:.1f}x)'
        self.speedLabel.setText(s)

        if self.monMQ.connected:
            msg = f"timescale {f:.1f}"
            self.monMQ.send_back_msg(msg)
        if hasattr(self, "logPlayer"):
            self.logPlayer.changeTimeScale(f)

    def connectButtonClicked(self):
        if self.msgThreadRunning:
            self.statusBar.showMessage ("Message thread is already running")
            return

        url = self.connectURL.text()
        if self.monMQ.bind(url):
            self.statusBar.showMessage("Connected successfully")
        else:
            self.statusBar.showMessage("Connection failed")

    def pauseButtonClicked(self):
        # if not self.msgThreadRunning:
        #     return  # no thread to pause yet

        if self.paused :
            self.paused = False
            self.pauseButton.setText("Pause")
            self.statusBar.showMessage ("Resumed")
            # self.pause_cond.notify()
            self.pauseLock.release()
        else:
            self.pauseLock.acquire()
            self.paused = True
            self.pauseButton.setText("Resume")
            self.statusBar.showMessage ("Paused")

    def dropButtonClicked(self):
        if not self.monMQ.connected: return
        if self.dropping :
            self.dropping = False
            self.dropButton.setText("Drop Msg")
            self.statusBar.showMessage ("Getting messages")
            self.monMQ.send_back_msg("restart")
        else:
            self.dropping = True
            self.dropButton.setText("Get Msg")
            self.statusBar.showMessage ("Dropping messages")
            self.monMQ.send_back_msg("stop")

    def playLogButtonClicked(self):
        res = QFileDialog.getOpenFileName(self)
        fName = res[0]
        if fName != "":
            timeScale = self.getTimeScale()
            self.statusBar.showMessage (f"Playing monitor log file {fName}")
            self.logPlayer = LogPlayer(self, fName, timeScale)
            self.logPlayer.start()


    @pyqtSlot(list)
    def setWidgetHandler(self, param):
        # assumes that param[0] == 'widget'
        w = self.getWidget(param[1])
        if w is None:
            print(f"widget {param[1]} not found")
            return
        verb = param[2]
        if verb == 'text':
            w.setText(param[3])
        elif verb == 'inc':
            v = int(w.text()) + int(param[3])
            w.setText(str(v))
        elif verb == 'value':
            # w.setValue(float(param[3]))
            w.setText(param[3])
        elif verb == 'table': # table item text
            r = int(param[3])
            c = int(param[4])
            item = w.item(r,c)
            if item is None:
                item = QTableWidgetItem()
                w.setItem(r,c,item)
            item.setText(param[5])
        elif verb == 'rows': # rows header_list
            row_headers = param[3:]
            w.setRowCount(len(row_headers))
            w.setVerticalHeaderLabels(row_headers)
        elif verb == 'columns': # columns header_list
            col_headers = param[3:]
            w.setRowCount(len(col_headers))
            w.setVerticalHeaderLabels(col_headers)
        elif verb == "timeScale":
            s = int(float(param[3]))
            w.setValue(s)
        else:
            print(f"widget {param} not found")
        # print (f"widget message done {param}")

    def closeEvent(self, event):
        self.msgThreadRunning = False
        self.logThreadRunning = False


class Viewport2D(QGraphicsView):
    def initViewport2D(self, parent):
        self.parent = parent
        self.scene = QGraphicsScene()
        parent.book = ObjBook(parent, self, self.scene)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setCacheMode(QGraphicsView.CacheNone)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        # self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # self.scene.setSceneRect(-35000, -9000, 170000, 140000)  # 금산 3공장 AGV

        self.offset_x = 0
        self.offset_y = 0
        self.invert_y = False

    # def mouseMoveEvent(self, e):  # event QMouseEvent
    def mousePressEvent(self, e):  # event QMouseEvent
        pos = self.mapToScene(e.pos())
        x = pos.x()
        y = pos.y()

        x -= self.offset_x
        y -= self.offset_y
        if self.invert_y:  y = -y

        msg = f"mouse pos = ({x:.0f}, {y:.0f})"
        self.parent.statusBar.showMessage(msg)
        super().mousePressEvent(e)

    def zoom(self, up=True, fine=False):
        delta = 0.25
        if fine:
            delta /= 10
        if up:
            s = 1 + delta
        else:
            s = 1 - delta
        self.scale(s, s)

    def wheelEvent(self, e):
        self.zoom(e.angleDelta().y() > 0, e.modifiers() == Qt.ControlModifier)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Up:
            up = True
        elif e.key() == Qt.Key_Down:
            up = False
        else:
            return
        self.zoom(up, e.modifiers() == Qt.ControlModifier)

    # this does not work somehow...ㅠㅠ
    # def event(self, e):
    #     if e.type() == QEvent.Gesture:
    #         return self.gestureEvent(e)
    #     return super().event(e)
    #
    # def gestureEvent(self, e):
    #     print("gesture event", e)
    #     print(f"scaleFactor = {e.scaleFactor()}")
    #     print(f"totalScaleFactor = {e.totalScaleFactor()}")


def run_monitor(uic, build_scene, connect_at_start=False, cfg_name=None):
    try:
        app = QApplication(sys.argv)
        window = SimAnimWindow(uic)
        build_scene(window.book, cfg_name)
        # buildScene_test1(window.book)
        # buildScene_TendemQueue(window.book)
        # buildScene_NYKim_Minifab(window.book)
        # buildScene_SmallOHT(window.book)
        # buildScene_AGV(window.book, cfg_name)
        # buildScene_Jobshop(window.book)
        # build_scene_Hanta1(window.book)
        # build_scene_Gumsan3(window.book)
        # build_DES_Tutorial_scene(window.book)

        # window.show()
        window.showMaximized()
        if connect_at_start:
            window.connectButtonClicked()
        sys.exit(app.exec_())
    except Exception as e:
        print ("Error occurred :", e)
        exit(1)

# # main program
# from FIM_Scene import buildScene_FIM
# from HantaScene import build_scene_KP3
# if __name__ == '__main__':
#     # pass
#     # run_monitor(cfg_name="AGV_KP2")
#     run_monitor("../FIM/FIM_Mon.ui", buildScene_FIM, False)
#     # run_monitor("../Fab/HantaShopMon.ui", build_scene_KP3, False, None)

