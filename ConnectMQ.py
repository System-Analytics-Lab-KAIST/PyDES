import time
import zmq
# import pynng
import threading
from PyQt5.QtCore import QThread
import queue
import json

MON_IP = '127.0.0.1'
MON_PORT = 5555
MON_URL = f"tcp://{MON_IP}:{MON_PORT}"

CON_IP = '127.0.0.1'
# CON_IP = '192.168.10.6'
single_URL = f"tcp://{CON_IP}:5563"
dual_URL = f"tcp://{CON_IP}:5564"
HS_URL = f"tcp://{CON_IP}:5565"
# G_hwm = 8000  # pynng에서는 8000이 max
# G_hwm = 1  # test with small buffer
# G_hwm = 1000000  # ZeroMQ에서는 매우 큰 값을 줄 수도 있음
G_timeout = 10000  # 10sec

class Pair_ZMQ(threading.Thread):
    def __init__(self, A_side=True, hook=None, dual_socket=True):
        super().__init__()
        self.A_side = A_side
        self.hook = hook
        self.handshake = False
        self.dual_socket = dual_socket
        self.in_socket = self.make_MQ_socket(make_context=True, timeout=G_timeout)
        if dual_socket:
            self.out_socket = self.make_MQ_socket(make_context=False, timeout=G_timeout)
        else:
            self.out_socket = self.in_socket

        if A_side:
            self.listen(self.in_socket, single_URL)
            if dual_socket:
                self.listen(self.out_socket, dual_URL)
        else:
            self.dial(self.out_socket, single_URL)
            if dual_socket:
                self.dial(self.in_socket, dual_URL)

        self.start()  # start a thread with self.run()

    @staticmethod
    def listen(socket, url):
        socket.bind(url)

    @staticmethod
    def dial(socket, url):
        socket.connect(url)

    def make_MQ_socket(self, make_context=False, timeout=0):
        if make_context:
            self.context = zmq.Context()
        socket = self.context.socket(zmq.PAIR)
        # socket.setsockopt(zmq.SNDHWM, G_hwm)
        # socket.setsockopt(zmq.RCVHWM, G_hwm)
        if timeout > 0:
            socket.setsockopt(zmq.RCVTIMEO, timeout)  # 10s까지 대기
        return socket

    def set_hook(self, hook):
        self.hook = hook

    def wait_handshake(self):
        while not self.handshake:
            time.sleep(0.001)

    def shake_hands(self):
        hs_socket = self.make_MQ_socket(make_context=False, timeout=0)
        if self.A_side:
            self.listen(hs_socket, HS_URL)
            hs_socket.send("A_side:ready".encode('utf-8'))
            reply = hs_socket.recv().decode('utf-8')
            print (reply)
        else:
            self.dial(hs_socket, HS_URL)
            req = hs_socket.recv().decode('utf-8')
            hs_socket.send("B_side:ready".encode('utf-8'))
            print (req)
        hs_socket.close()
        self.handshake = True


    def send_stop(self):
        self.send_json({'message': 'stop'})

    def close_sockets(self):
        self.in_socket.close()
        if self.dual_socket:
            self.out_socket.close()

    def run(self):
        # confirm connection is ready
        self.shake_hands()
        self.timeout_sleeping = False
        while True:
            try:
                # print ("start recv_json")
                d = self.recv_json()
                self.timeout_sleeping = False
                # print ("end recv_json:", len(d))
                if d['message'] == 'stop':
                    break
                if self.hook:
                    d["Clock_recv"] = time.time()  # message 수령 시간
                    self.hook(d)
                else:
                    print(json.dumps(d, indent=4))
                # events = self.socket.getsockopt(zmq.EVENTS)
                # if events & zmq.POLLIN:
                #     print ("message is waiting in ZeroMQ")
            except zmq.Again as e:
                print ("Timeout error while waiting for message: ", e)
                self.timeout_sleeping = True
                time.sleep(0.01)  # 메시지가 없을 경우 짧은 대기
            # except Exception as e:
            #     print ("Error (timeout) occurred while receiving message: ", e)
            #     exit(990)
        self.send_stop()  # stop counterpart, too
        self.close_sockets()

    def send_string(self, text):
        self.out_socket.send_string(text)  # send_string, recv_string do encoding, decoding

    def recv_string(self):
        return self.in_socket.recv_string()

    def send_json(self, obj):
        # self.socket.send_json(obj)    # 상대편에서 send_json, recv_json 쓸 경우
        text = json.dumps(obj)   # 상대편에서 send_json, recv_json 쓰지 않을 경우
        self.out_socket.send_string(text)
        if self.timeout_sleeping:
            print("send to ACS:", text)

    def recv_json(self):
        # return self.socket.recv_json()   # 상대편에서 send_json, recv_json 쓸 경우
        text = self.in_socket.recv_string()  # 상대편에서 send_json, recv_json 쓰지 않을 경우
        return json.loads(text)




# class Pair_NNG(Pair_ZMQ):

#     def make_MQ_socket(self, make_context=False):
#         nng_socket = pynng.Pair0()
#         # nng_socket.send_buffer_size = G_hwm
#         # nng_socket.recv_buffer_size = G_hwm
#         return nng_socket


#     @staticmethod
#     def listen(socket, url):
#         socket.listen(url)

#     @staticmethod
#     def dial(socket, url):
#         socket.dial(url, block=False)

#     def send_string(self, text):
#         self.out_socket.send(text.encode('utf-8'))

#     def recv_string(self):
#         return self.in_socket.recv().decode('utf-8')

#     def send_json(self, obj):
#         text = json.dumps(obj).encode('utf-8')
#         self.out_socket.send(text)

#     def recv_json(self):
#         text = self.in_socket.recv().decode('utf-8')
#         return json.loads(text)


Pair_MQ = Pair_ZMQ
# Pair_MQ = Pair_NNG
G_dual_socket = True
# G_dual_socket = False

################################################################################
# make sure connection is synchronized
################################################################################

def syncConnect(context, url, bind=True):
    print("Waiting for connection")
    if bind:
        sync_socket = context.socket(zmq.REP)
        sync_socket.bind(url + "1")
        sync_socket.recv()
        sync_socket.send(b'')
    else:
        sync_socket = context.socket(zmq.REQ)
        sync_socket.connect(url + "1")
        sync_socket.send(b'')
        sync_socket.recv()
    sync_socket.close()
    print("Connection synced")



################################################################################
# connect to zmq from Simulation (server) side
################################################################################
class SimulationMQ:
    def __init__(self):
        self.pubSocket = None
        self.subSocket = None
        # self.pairSocket = None  # this is for dual fed sim
        self.url = MON_URL
        self.stopped = False
        self.connected = False
        self.context = zmq.Context()

    @staticmethod
    def purgeSocket(socket):
        try:
            n = 0
            while True:
                msg = socket.recv_string(flags=zmq.NOBLOCK)
                print (msg)
                n += 1
        except zmq.Again:
            return n


    def setStopSending(self, s):
        self.stopped = s

    def send_msg_raw(self, msg):
        if self.connected:
            self.pubSocket.send_string(msg)
        # print(msg)
        # pass

    def send_msg(self, msg):
        if self.connected and not self.stopped:
            self.pubSocket.send_string(msg)
        # print(msg)
        # pass

    def check_mon_msg(self):
        try:
            msg = self.subSocket.recv_string(flags=zmq.NOBLOCK)
            return msg
        except zmq.Again:
            return None


    def connect(self, url=None, sync=True):  # hwm=1000000*4
        if url is not None:
            self.url = url
        self.pubSocket = self.context.socket(zmq.PUB)
        hwm = 1100000
        self.pubSocket.setsockopt(zmq.SNDHWM, hwm)
        self.pubSocket.setsockopt(zmq.RCVHWM, hwm)
        # self.pubSocket.bind(self.url)
        self.pubSocket.connect(self.url)

        # exchange sync messages
        if sync:
            syncConnect(self.context, self.url, bind=True)

        self.subSocket = self.context.socket(zmq.SUB)
        self.subSocket.setsockopt(zmq.SNDHWM, hwm)
        self.subSocket.setsockopt(zmq.RCVHWM, hwm)
        self.subSocket.connect(self.url + "2")

        filter = ""  # get every message
        self.subSocket.setsockopt_string(zmq.SUBSCRIBE, filter)

        print("Connected successfully")
        self.connected = True


    # def connectPair(self, url=None, bind=True):
    #     if url is not None:
    #         self.url = url
    #     self.pairSocket = self.context.socket(zmq.PAIR)
    #     url = self.url + "3"
    #     if bind:
    #         self.pairSocket.bind(url)
    #     else:
    #         self.pairSocket.connect(url)
    #     print("Connected to pair socket.  Waiting for partner")
    #
    # def sendPairZmq(self, msg):
    #     self.pairSocket.send_string(msg)
    #
    # def checkPairZmq(self):
    #     try:
    #         return self.pairSocket.recv_string(flags=zmq.NOBLOCK)
    #     except zmq.Again:
    #         return None
    #
    # def waitPairZmq(self):
    #     return self.pairSocket.recv_string()


# class MonitorMQ(threading.Thread):
class MonitorMQ(QThread):
    def __init__(self, mainWnd):
        super().__init__()
        self.subSocket = None
        self.pubSocket = None
        self.url = MON_URL
        self.connected = False
        self.mainWnd = mainWnd
        self.book = mainWnd.book
        self.context = zmq.Context()

    def send_back_msg(self, msg):
        print("Message to simulator : ", msg)
        self.pubSocket.send_string(msg)

    ################################################################################
    # mainWnd : main window of monitor
    # mainWnd.book : object book
    # mainWnd.msgThreadRunning : False if mainWnd is closed, True otherwise
    # mainWnd.paused : True if pause button is pressed
    ################################################################################
    def run(self):
        print ("Message thread started")
        mid = 0
        objBook = self.mainWnd.book
        while self.mainWnd.msgThreadRunning:
            mid += 1
            msg = self.subSocket.recv_string()
            # print(msg)
            objBook.processMsg(msg)  # process each message
            if mid == 100:
                mid = 0
                time.sleep(0.0001)
                if self.mainWnd.paused:
                    self.mainWnd.pauseLock.acquire()
                    self.mainWnd.pauseLock.release()


    ################################################################################
    # mainWnd : main window of monitor
    # connect simulation server and run message thread
    ################################################################################
    def bind(self, url=None):
        if url is None:
            self.url = url
        # self.subSocket = self.context.socket(zmq.SUB)
        # self.subSocket.connect(url)
        self.subSocket = self.context.socket(zmq.SUB)
        hwm = 1100000
        self.subSocket.setsockopt(zmq.SNDHWM, hwm)
        self.subSocket.setsockopt(zmq.RCVHWM, hwm)
        self.subSocket.bind(url)

        # filter = "color"  # get messages starting with 'P'
        filter = ""  # get every message
        self.subSocket.setsockopt_string(zmq.SUBSCRIBE, filter)

        self.mainWnd.msgThreadRunning = True
        self.connected = True
        self.start()

        # exchange sync messages
        if self.mainWnd.syncServer.isChecked():
            self.mainWnd.statusBar.showMessage("Waiting for server to connect at "+url)
            syncConnect(self.context, self.url, bind=False)

        self.pubSocket = self.context.socket(zmq.PUB)
        self.pubSocket.setsockopt(zmq.SNDHWM, hwm)
        self.pubSocket.setsockopt(zmq.RCVHWM, hwm)
        self.pubSocket.bind(url + "2")

        return True




