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
    CM = enum.auto()
    PM = enum.auto()



class MyQueue(list):
    def __init__(self, name, selector=None, emq=None, log=None, widget=None):
        self.name = name
        if selector is None:
            selector = self.fifo_selector
        self.selector = selector
        self.emq = emq
        self.log = log
        if log is not None:
            TableStat.register("Queue", self)
        self.widget = widget

    def get_stat_series(self, newT):
        return self.log.get_stat_series(newT, self.name)

    def reset_stat(self, t):
        self.log.reset(t)

    def qsize(self):
        return len(self)

    def mon_log(self):
        if self.emq:
            s = self.qsize()
            self.emq.sendMonMsg(f"queue {self.name} {s}")
            if self.log:
                self.log.stateChange(self.emq.Tnow, s)
                qa = self.log.timeAvg(self.emq.Tnow)
                if self.widget:
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



class MyPQ:
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


class BatchMean(list):
    bm_list = []

    @classmethod
    def register(cls, obj):
        # list.append(cls.bm_list, obj)
        cls.bm_list.append(obj)

    @classmethod
    def save_batch_all(cls, t=None):
        for bm in cls.bm_list:
            bm.save_batch(t)

    def __init__(self):
        super().__init__()
        BatchMean.register(self)

    def batch_len(self): return len(self)
    
    def report(self, s=0):
        A = np.array(self[s:])
        avg = A.mean()
        serr = A.std() / np.sqrt(len(A))
        return avg, serr, len(A)

    def __str__(self):
        avg, serr, n = self.report(s=1)
        return f"{avg:.3f} (+/- {serr:.3f}, n={n})"

    def get_batch_mean(self, t=None):  return None  # to be implemented in subclass

    # assumes: get_batch_mean(t) is defined
    def save_batch(self, t=None):
        m = self.get_batch_mean(t)
        if m is not None:
            list.append(self,m)


from collections import defaultdict
class SingleCounter(BatchMean):
    def __init__(self):
        super().__init__()
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

    def get_batch_mean(self, t):  
        m = self.value
        self.value = 0
        return m


class MultiCounter(defaultdict):
    def __init__(self):
        super().__init__(int)
        self.batch_mean = defaultdict(BatchMean)

    def saveBatch(self, t=None):
        for key in self.keys():         
            self.batch_mean[key].append(self[key])
            self[key] = 0

    def get_batch_mean(self, key):
        return self.batch_mean[key].report()


class Accumulator(BatchMean):
    def __init__(self, history=False):
        super().__init__()
        self.resetHistory(history)
        self.reset()

    def reset_stat(self, t=None):
        self.reset()

    def get_stat_series(self, t, name=None):
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
            return f"{avg:.3f} (+/- {serr:.3f}, n={n})"
        else:
            return BatchMean.__str__(self)

    def sum(self):
        return self.X

    def get_batch_mean(self, t):  
        m = self.avg()
        self.reset()
        return m

    

class TimeAverage(BatchMean):
    def __init__(self, t=0, v=0, history=False): # history가 true면 적분 그래프 표시
        super().__init__()
        self.reset(t, v)
        self.resetHistory(history, t, v)
        self.drawChart = self.plotHistory  # old name

    def get_result(self):
        if self.batch_len() <= 1:
            avg = self.timeAvg(self.prevT)
            serr, n = 0, 1
        else:
            avg, serr, n = self.report(s=1)
        return avg, serr, n


    def __str__(self):
        avg, serr, n = self.get_result()
        s = f"{avg:.3f}"
        if n > 1:
            s += f" (+/- {serr:.3f}, n={n})"
        return s

    def reset(self, t, v=None):
        self.initT = t
        self.prevT = t
        if v is not None:
            self.prevV = v
        self.sum = 0.0
        self.XX = 0.0
        self.min = infinity
        self.max = -infinity

    def get_batch_mean(self, t):  
        m = self.timeAvg(t)
        self.reset(t)
        return m

    def timeSinceReset(self, t):
        return t - self.initT

    def get_stat_series(self, t, name=None):
        data = [self.timeAvg(t), self.min, self.max]
        return pd.Series(data, index=["avg", "min", "max"], name=name)

    def resetHistory(self, history, t=0, v=0):
        self.history = history
        if self.history:
            self.histT = [t]
            self.histV = [v]

    def timeAvg(self, T, std=False):
        dT = T - self.initT
        if dT <= 0.0:  return 0, 0 if std else 0
        self.stateChange(T, self.prevV)  # maintain current state
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


    def stateChange(self, t, v):
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

    def stateInc(self, t, dv=1):
        self.stateChange(t, self.prevV+dv)

    def plotHistory(self, ax=None, label=None, t=None):
        if t: self.stateInc(t,0)
        if not self.history: return
        if ax is None:
            fig, ax = plt.subplots() # (figsize=(13,7))
        ax.step(self.histT, self.histV, where='post', label=label)
        return ax


class TableStat:
    book = {}

    # this is assumed to have the following methods
    # - get_stat_series(t): returns a pd.Series
    # - reset_stat(t) : resets stat of this

    @classmethod
    def register(cls, table_name, this):
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
    def to_excel_sheet(cls, table_name, writer, T):
        L = cls.book[table_name]
        if len(L) == 2 and isinstance(L[1], pd.DataFrame):
            df = L[1]
        else:
            L_stat = [ent.get_stat_series(T) for ent in L]
            if isinstance(L_stat[0], pd.DataFrame):
                df = pd.concat(L_stat, axis=0)
            else:
                df = pd.concat(L_stat, axis=1)
                df.columns = [stat.name for stat in L_stat]
                df = df.T
        df.to_excel(excel_writer=writer, sheet_name=table_name)
        return df


    @classmethod
    def to_excel_all(cls, fn, T):
        def write_excel(fn):
            with pd.ExcelWriter(fn, mode="w", engine='openpyxl') as writer:
                for table_name in cls.book.keys():
                    df = cls.to_excel_sheet(table_name, writer, T)
                    df_list.append(df)

        df_list = []
        try:
            write_excel(fn)
        except Exception as e:
            print (e)
            print (f"{fn} is already open.  output to 'Stat.xlsx' instead ")
            write_excel("Stat.xlsx")
        return df_list
    
    @classmethod
    # table하나에 stat entity 하나 있고, 시간에 따라 여러 record가 만들어지는 경우
    def append_and_reset(cls, table_name, T):
        L = cls.book[table_name]
        ent = L[0]
        S = ent.get_stat_series(T)
        S_df = pd.DataFrame(data=[S], index=[S.name], columns=S.index)
        S_df["@Time"] = T
        if len(L) == 1:  # df is not made yet
            L.append(S_df)
        else:  # df already exists
            df = L[1]
            L[1] = pd.concat([df, S_df])
        ent.reset_stat(T)



class StateTime:
    def __init__(self, table_name, row_name, states, t=0, init_state=-1):
        self.S = pd.Series(index=states, name=row_name)
        self.prevS = states[init_state]
        self.reset_stat(t)
        TableStat.register(table_name, self)

    def reset_stat(self, t):
        self.S[:] = 0.0
        self.initT = t
        self.prevT = t

    def get_stat_series(self, newT):
        self.setState(self.prevS, newT)
        return self.S

    def time_percentage(self, newT):
        self.setState(self.prevS, newT)
        if newT <= self.initT:
            return self.S
        return self.S / (newT - self.initT) * 100

    def setState(self, newS, newT):
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

    def to_excel(self, writer, sheet=None):
        # self.S.to_excel(writer, sheet_name=sheet)
        self.S.to_excel(excel_writer=writer)


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
