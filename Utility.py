import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
import pandas as pd

import time
import math
import heapq
import matplotlib.pyplot as plt
import pickle
import os.path
import enum

infinity = np.inf


# so that State can be used as a key of dictionary
class HashableEnum(enum.Enum):
    def __eq__(self, other):
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


# example of state definition
class MState(HashableEnum):
    IDLE = enum.auto()
    WORK = enum.auto()
    SETUP = enum.auto()
    CM = enum.auto()  # corrective maintenance = repair
    PM = enum.auto()  # preventive maintenance


class PyDES_Queue(list):
    def __init__(self, name, emq=None, selector=None, log=True, widget=None):
        # super().__init__()
        self.name = name
        if selector is None:
            selector = self.fifo_selector
        self.selector = selector
        self.emq = emq
        if log != False:
            TableStat.register("Queue", self)
            if isinstance(log, TimeAverage): 
                self.log = log
            else:  # create
                self.log = TimeAverage(name=name)
        self.widget = widget

    def get_stat_series(self):
        return self.log.get_stat_series(name=self.name)

    def reset_stat(self):
        self.log.reset()

    def qsize(self):
        return len(self)

    def mon_log(self):
        if self.emq is not None:
            s = self.qsize()
            self.emq.sendMonMsg(f"queue {self.name} {s}")
            if self.log is not None:
                self.log.stateChange(s)
                if self.widget:
                    qa = self.log.timeAvg()
                    self.emq.sendMonMsg(f"widget {self.widget} value {qa:.3f}")

    def enqueue(self, e):
        self.append(e)
        self.mon_log()

    def dequeue(self):
        try:
            idx = self.selector(self)
            e = self.pop(idx)
            self.mon_log()
            return e
        except IndexError:
            return None

    @staticmethod
    def lifo_selector(L):
        return len(L) - 1

    @staticmethod
    def fifo_selector(L):
        return 0


class PyDES_PQ:
    def __init__(self):
        self.heap = []

    def clear(self):
        self.heap.clear()

    def qsize(self):
        return len(self.heap)

    def len(self):
        return len(self.heap)

    def __len__(self):
        return len(self.heap)

    def empty(self):
        return len(self.heap) == 0

    def put(self, obj):
        heapq.heappush(self.heap, obj)

    def get(self):
        return heapq.heappop(self.heap)

    def peep(self):
        if self.empty(): return None
        return self.heap[0]

    def peep_last(self):
        if self.empty(): return None
        return self.heap[-1]
        # max_i = 0
        # for i in range(1, len(self.heap)):
        #     if self.heap[i] > self.heap[max_i]:
        #         max_i = i
        # n = len(self.heap)
        # return self.heap[max_i]

    def heapify(self):
        heapq.heapify(self.heap)


class PyDES_PQwLog(PyDES_PQ):
    def __init__(self, name, emq=None, log=True):
        super().__init__()
        self.name = name
        self.emq = emq
        if log != False:
            TableStat.register("PQ", self)
            if isinstance(log, TimeAverage):
                self.log = log
            else:  
                self.log = TimeAverage(name=name)

    def get_stat_series(self):
        return self.log.get_stat_series(name=self.name)

    def reset_stat(self):
        self.log.reset()

    def put(self, obj):
        heapq.heappush(self.heap, obj)
        self.log.stateChange(len(self))

    def get(self):
        e = heapq.heappop(self.heap)
        self.log.stateChange(len(self))
        return e


class PyDES_MultiQueue_Abstract(dict):
    def __init__(self, name, emq):
        super().__init__()
        self.emq = emq
        self.name = name

    def make_queue(self, key):
        pass
    
    def __missing__(self, key):  # self[key]가 없을 경우, item 생성
        value = self[key] = self.make_queue(key)
        return value

    def qsize(self):
        s = 0
        for q in self.values():
            s += q.qsize()
        return s


class PyDES_MultiQueue(PyDES_MultiQueue_Abstract):
    def __init__(self, name, emq, selector=None):
        super().__init__(name, emq)
        self.selector = selector

    def make_queue(self, key):
        return PyDES_Queue(f"{self.name}_{key}", self.emq, self.selector, log=True)


class PyDES_MultiPQ(PyDES_MultiQueue_Abstract):
    def make_queue(self, key):
        return PyDES_PQwLog(f"{self.name}_{key}", self.emq, log=True)


class BatchMean(list):
    @classmethod
    def init_list(cls):
        cls.bm_list = []

    @classmethod
    def register(cls, obj):
        if "bm_list" not in dir(cls):
            cls.init_list()
        # list.append(cls.bm_list, obj)
        cls.bm_list.append(obj)

    @classmethod
    def save_batch_all(cls):
        for bm in cls.bm_list:
            bm.save_batch()

    @classmethod
    def get_batch_df(cls, fn=None):
        L = {bm.name: bm for bm in cls.bm_list}
        df = pd.DataFrame(L)
        if fn is not None:
            with pd.ExcelWriter(fn, mode="w", engine='openpyxl') as writer:
                df.to_excel(excel_writer=writer)
        return df

    def __init__(self, name=""):
        super().__init__()
        self.name = name
        BatchMean.register(self)

    def batch_len(self): return len(self)
    
    def report(self, s=0):
        A = np.array(self[s:])
        avg = A.mean()
        serr = A.std() / np.sqrt(len(A))
        return avg, serr, len(A)

    def __str__(self):
        avg, serr, n = self.report(s=1)
        return f"{self.name} {avg:.3f} (+/- {serr:.3f}, n={n})"

    def get_batch_mean(self):  return None  # to be implemented in subclass

    # assumes: get_batch_mean() is defined
    def save_batch(self):
        m = self.get_batch_mean()
        if m is not None:
            list.append(self,m)


from collections import defaultdict
class SingleCounter(BatchMean):
    def __init__(self, name=""):
        super().__init__(name)
        self.value = 0

    def set(self, n):
        self.value = n

    def inc(self, n=1):
        self.value += n

    def __add__(self, other):
        self.value += other
        return self

    def __repr__(self):
        return f"Counter({self.value})"

    def get_batch_mean(self):  
        m = self.value
        self.value = 0
        return m


class MultiCounter(defaultdict):
    def __init__(self):
        super().__init__(int)
        self.batch_mean = defaultdict(BatchMean)

    def save_batch(self):
        for key in self.keys():         
            self.batch_mean[key].append(self[key])
            self[key] = 0

    def get_batch_mean(self, key):
        return self.batch_mean[key].report()


class Accumulator(BatchMean):
    def __init__(self, name="", history=False):
        super().__init__(name)
        self.resetHistory(history)
        self.reset()

    def reset_stat(self):
        self.reset()

    def get_stat_series(self, name=None):
        if name is None: name = self.name
        data = [self.n, self.avg(), self.std(), self.min, self.max]
        return pd.Series(data, index=["n", "avg", "std", "min", "max"], name=name)

    def reset(self):
        self.n = 0
        self.X = 0.0
        self.XX = 0.0
        self.min = infinity
        self.max = -infinity

    def resetHistory(self, history):
        self.history = history
        if self.history:
            self.histX = []

    def append(self, x):
        if self.history:
            self.histX.append(x)
        self.n += 1
        self.X += x
        self.XX += x*x
        if x < self.min:
            self.min = x
        if x > self.max:
            self.max = x

    def avg(self):
        if self.n == 0: return 0.0
        return self.X / self.n

    def var(self):
        if self.n == 0: return 0.0
        v = self.XX / self.n - self.avg()**2
        return max(v,0)

    def std(self):
        return math.sqrt(self.var())

    def __str__(self):
        if self.batch_len() <= 1:
            avg, serr, n = self.avg(), self.std(), self.n
            serr /= np.sqrt(n)
            return f"{self.name} {avg:.3f} (+/- {serr:.3f}, n={n})"
        else:
            return BatchMean.__str__(self)

    def sum(self):
        return self.X

    def get_batch_mean(self):  
        m = self.avg()
        self.reset()
        return m

def get_sim_clock(clock=None):
    if clock is None:
        from DES import EvtMsgQueue
        clock = EvtMsgQueue.sim_clock
    return clock

class TimeAverage(BatchMean):
    def __init__(self, name="", clock=None, v=0, history=False): # history가 true면 적분 그래프 표시
        super().__init__(name)
        self.clock = get_sim_clock(clock)
        self.reset(v)
        self.resetHistory(history, v)
        self.drawChart = self.plotHistory  # old name

    def get_result(self):
        if self.batch_len() <= 1:
            avg = self.timeAvg()
            serr, n = 0, 1
        else:
            avg, serr, n = self.report(s=1)
        return avg, serr, n


    def __str__(self):
        avg, serr, n = self.get_result()
        s = f"{self.name} {avg:.3f}"
        if n > 1:
            s += f" (+/- {serr:.3f}, n={n})"
        return s

    def reset(self, v=None):
        t = self.clock()
        self.initT = t
        self.prevT = t
        if v is not None:
            self.prevV = v
        self.sum = 0.0
        self.XX = 0.0
        self.min = infinity
        self.max = -infinity

    def get_batch_mean(self):
        m = self.timeAvg()
        self.reset()
        return m

    def get_stat_series(self, name=None):
        if name is None: name = self.name        
        data = [self.timeAvg(), self.min, self.max]
        return pd.Series(data, index=["avg", "min", "max"], name=name)

    def resetHistory(self, history, v=0):
        self.history = history
        if self.history:
            self.histT = [self.clock()]
            self.histV = [v]

    def timeAvg(self, std=False):
        dT = self.clock() - self.initT
        if dT <= 0.0:  return 0, 0 if std else 0
        self.stateChange(self.prevV)  # maintain current state
        avg = self.sum / dT
        if std:
            var = self.XX/dT - avg*avg
            if var < 0.0:  # due to numerical error
                # print (f"negative variance {var}")
                std = 0.0
            else:
                std = np.sqrt(var)
            return avg, std
        else:
            return avg


    def stateChange(self, v):
        t = self.clock()
        if self.history:
            if t == self.prevT:
                self.histV[-1] = v
            else:
                self.histT.append(t)
                self.histV.append(v)
        if t > self.initT:
            if v < self.min:
                self.min = v
            if v > self.max:
                self.max = v
        dt = t - self.prevT
        self.sum += self.prevV * dt
        self.XX += self.prevV * self.prevV * dt
        self.prevT = t
        self.prevV = v

    def stateInc(self, dv=1):
        self.stateChange(self.prevV+dv)

    def plotHistory(self, ax=None, label=None):
        self.stateInc(0)
        if not self.history: return
        if ax is None:
            fig, ax = plt.subplots() # (figsize=(13,7))
        ax.step(self.histT, self.histV, where='post', label=label)
        return ax


class TableStat:
    @classmethod
    def init_book(cls):
        cls.book = {}

    # this is assumed to have the following methods
    # - get_stat_series(): returns a pd.Series
    # - reset_stat() : resets stat of this

    @classmethod
    def register(cls, table_name, this):
        if "book" not in dir(cls):
            cls.init_book()
        try:
            cls.book[table_name].append(this)
        except KeyError:
            cls.book[table_name] = [this]

    @classmethod
    def reset_all(cls, t):
        for L in cls.book.values():
            for this in L:
                this.reset_stat(t)

    @classmethod
    def to_excel_sheet(cls, table_name, writer):
        L = cls.book[table_name]
        if len(L) == 2 and isinstance(L[1], pd.DataFrame):
            df = L[1]
        else:
            L_stat = [ent.get_stat_series() for ent in L]
            if isinstance(L_stat[0], pd.DataFrame):
                df = pd.concat(L_stat, axis=0)
            else:
                df = pd.concat(L_stat, axis=1)
                df.columns = [stat.name for stat in L_stat]
                df = df.T
        df.to_excel(excel_writer=writer, sheet_name=table_name)
        return df


    @classmethod
    def to_excel_all(cls, fn):
        def write_excel(fn):
            with pd.ExcelWriter(fn, mode="w", engine='openpyxl') as writer:
                for table_name in cls.book.keys():
                    df = cls.to_excel_sheet(table_name, writer)
                    df_list.append(df)

        df_list = []
        try:
            write_excel(fn)
        except Exception as e:
            print (e)
            fn2 = fn[:-5]+"(2).xlsx"
            print (f"{fn} is already open.  output to {fn2} instead ")
            write_excel(fn2)
        return df_list
    
    @classmethod
    # table하나에 stat entity 하나 있고, 시간에 따라 여러 record가 만들어지는 경우
    def append_and_reset(cls, table_name, T):
        L = cls.book[table_name]
        ent = L[0]
        S = ent.get_stat_series()
        S_df = pd.DataFrame(data=[S], index=[S.name], columns=S.index)
        S_df["@Time"] = T  # save the time of record
        if len(L) == 1:  # df is not made yet
            L.append(S_df)
        else:  # df already exists
            df = L[1]
            L[1] = pd.concat([df, S_df])
        ent.reset_stat()



class StateTime:
    def __init__(self, table_name, row_name, states, clock=None, init_state=-1):
        self.S = pd.Series(index=states, name=row_name)
        self.prevS = states[init_state]
        self.clock = get_sim_clock(clock)
        self.reset_stat()
        TableStat.register(table_name, self)

    def reset_stat(self):
        t = self.clock()
        self.S[:] = 0.0
        self.initT = t
        self.prevT = t

    def get_stat_series(self):
        self.setState(self.prevS)
        return self.S

    def time_percentage(self):
        newT = self.clock()
        self.setState(self.prevS)
        if newT <= self.initT:
            return self.S
        return self.S / (newT - self.initT) * 100

    def setState(self, newS):
        newT = self.clock()
        dt = newT - self.prevT
        self.S[self.prevS] += dt
        self.prevT = newT
        self.prevS = newS

    def addCount(self, col, n=1):
        self.S[col] += n


class JobLogTable:
    def __init__(self, states=None, record=True):
        self.S = pd.DataFrame(columns=states)
        self.record = record

    def setAttribute(self, jobid, attribute, value):
        if self.record:
            self.S.loc[jobid, attribute] = value

    def getAttribute(self, jobid, attribute):
        return self.S.loc[jobid, attribute]

    def to_excel(self, writer, sheet="Sheet1"):
        self.S.to_excel(excel_writer=writer, sheet_name=sheet)
 

class PyDES_DefaultDict(dict):
    def __init__(self, value_cls):
        super().__init__()
        self.value_cls = value_cls  # value에 사용할 class명 

    def __missing__(self, key): 
        # self[key]가 없을 경우, value 생성
        # key를 인자로 사용할 수 있음
        value = self[key] = self.value_cls()
        return value


def rotate(dx, dy, x, y, ox=0, oy=0):
    d = math.sqrt(dx * dx + dy * dy)
    assert d > 0, "utility.rotate, d == 0"
    c = dx / d
    s = dy / d
    xp = c*x - s*y + ox
    yp = s*x + c*y + oy
    return xp, yp


def pickle_dump(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(pkl_fn, src_fn=None):
    try:
        if src_fn and os.path.exists(src_fn) and os.path.exists(pkl_fn):
            src_time = os.path.getmtime(src_fn)
            pkl_time = os.path.getmtime(pkl_fn)
            if src_time > pkl_time:
                return None
        with open(pkl_fn, 'rb') as f:
            obj = pickle.load(f)
            return obj
    except:
        return None


def pickle_dump_graph(g, fn):
    # sys.setrecursionlimit(10 ** 6)
    # # print(sys.getrecursionlimit())
    g.pre_pickle()  # Edge --> Vertex link 끊어주기
    pickle_dump(g, fn)
    g.post_pickle()

def pickle_load_graph(pkl_fn, src_fn=None):
    g = pickle_load(pkl_fn, src_fn)
    if g:
        g.post_pickle()  # Edge --> Vertex link 연결해주기
    return g


# Colors = ["white", "black", "lightgray", "blue", "royalblue", "lime", "blueviolet", "tan",
#           "orange", "red", "pink", "khaki", "gray", "darkgray", "darkgreen", "cyan", "lightgreen", "lightblue"]
Colors = ["white", "black",
          "gray", "lightgray",
          "red", "pink",
          "darkgreen", "lightgreen",
          # "orangered", "lightsalmon",
          "darkred", "lightsalmon",
          "blue", "lightblue",
          "darkviolet", "violet",
          "cyan", "lightcyan",
          "gold", "lightyellow"]

Colors_Max = len(Colors) - 1

# M_Palette  = {
#     State.WHITE: mcolors.CSS4_COLORS["white"],
#     State.NONE : mcolors.CSS4_COLORS["black"],
#     State.IDLE : mcolors.CSS4_COLORS["lightgray"],
#     State.UNLOADING: mcolors.CSS4_COLORS["blue"],
#     State.LOADING: mcolors.CSS4_COLORS["royalblue"],
#     State.WORKING: mcolors.CSS4_COLORS["lime"],
#     State.SETUP: mcolors.CSS4_COLORS["blueviolet"],
#     State.PM: mcolors.CSS4_COLORS["tan"],
#     State.DOWN: mcolors.CSS4_COLORS["black"],
#     State.WAIT_CART_IN: mcolors.CSS4_COLORS["red"],
#     State.WAIT_CART_OUT: mcolors.CSS4_COLORS["pink"],
#     State.REPLACING: mcolors.CSS4_COLORS["khaki"],
#     State.PAUSE: mcolors.CSS4_COLORS["gray"],
#     State.REST: mcolors.CSS4_COLORS["darkgray"],
#     State.MEETING: mcolors.CSS4_COLORS["darkgreen"],
#     # more colors : lightcoral, tomato, darkgreen, ... in css4 colors
# }


class SceneObj:
    def __init__(self):
        self.mon_obj = None
        self.rot = 0.0
        self.scale = 1.0
        self.x, self.y = 0, 0

    def makeSceneObj(self):
        return self.mon_obj


    def placeSceneObj(self, mo_book, x=0, y=0, rot_deg=0, scale=1):
        self.rot = rot_deg
        self.scale = scale
        self.x = x
        self.y = y

        if mo_book is None: return
        ml = self.makeSceneObj()
        if ml is None: return
        if not isinstance(ml, list):
            ml = [ml]
        for m in ml:
            if x != 0 or y != 0:
                m.moveSignalHandler([None, None, x, y])
            if rot_deg != 0:
                m.rotateSignalHandler([None, None, rot_deg])
            if scale != 1:
                m.scaleSignalHandler([None, None, scale])
            mo_book.append(m)

    # returns local coordinate of box corner  (x, y, wx, wy)
    def _get_pos_size_(component):
        pass


    # returns globa coordinate of box center
    def get_component_loc(self, component):
        _, x, y, wx, wy = self._get_pos_size_(component)
        cx, cy = x+wx/2, y+wx/2
        if self.rot != 0:
            t = self.rot * np.pi / 180
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            cx, cy = cx * cos_t - cy * sin_t, cx * sin_t + cy * cos_t
        if self.scale != 1:
            s = self.scale
            cx, cy = s*cx, s*cy
            wx, wy = s*wx, s*wy
        return cx+self.x, cy+self.y, wx, wy


class LogWriter:
    def __init__(self):
        self.open()

    def open(self, screen=True, filename=None):
        self.screen = screen
        if filename:
            dfn = filename
            # dfn = f"{G_param['data_folder']}/{filename}"
            self.logfile = open(dfn, 'w')
        else:
            self.logfile = None

    def raw_write(self, text, screen, logfile):
        if screen:
            print(text)
        if logfile:
            self.logfile.write(text)
            self.logfile.write('\n')

    def write(self, text):
        self.raw_write(text, self.screen, self.logfile)

    def warn(self, text):
        self.raw_write(text, True, self.logfile)

    def screen_only(self, text):
        self.raw_write(text, True, False)

    def close(self):
        if self.logfile:
            self.logfile.close()
            self.logfile = None

logger = LogWriter()


def circular_position(nx, ny, ccw, dx, dy, xp, yp, k):
    xk, yk = xp, yp
    if ccw:
        if k >= 2 * ny + nx - 3:
            xk -= dx
        elif k >= ny + nx - 2:
            yk -= dy
        elif k >= ny - 1:
            xk += dx
        else:
            yk += dy
    else:  # cw
        if k >= 2 * nx + ny - 3:
            yk -= dy
        elif k >= ny + nx - 2:
            xk -= dx
        elif k >= nx - 1:
            yk += dy
        else:
            xk += dx

    return xk, yk


def init_stat_class_variables():
    BatchMean.init_list()
    TableStat.init_book()
    