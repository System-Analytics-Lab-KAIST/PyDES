# from queue import *
import time
from ConnectMQ import *
import math
from Utility import *
from InputStat import RndNumGen
import pickle
import sys
import inspect
from itertools import count
from SimMonMainUI import run_monitor
import random
import torch
import numpy as np

import multiprocessing as mp
# import torch.multiprocessing as mp
# from concurrent.futures import ProcessPoolExecutor

from jax_backend import resolve, run_parallel, run_multiple_jax  # JAX backend import, (not JAX directly


# single federation version for simplicity
class Environment:
    current_env = None

    def __init__(self, name, emq=None):
        if emq is None:
            emq = EvtMsgQueue(name)
        Environment.current_env = self
        self.emq = emq
        emq.env = self

        self.name = name
        self.config = None  # experimental condifuration dict
        self.main_model = None
        self.p_models = []
        self.bm_list = []  # BatchMean instances
        self.table_stat = TableStat()
        self.job_log = JobLogTable(record=True)
        self.result = None

    def clock(self):  return self.emq.Tnow
    def save_result(self, res): self.result = res

    def config_item(self, key):
        return self.config[key] if self.config is not None else None

    def record_job_log(self, id, s, v):
        self.job_log.setAttribute(id, s, v)

    def get_job_log(self, id, a):
        return self.job_log.getAttribute(id, a)  

    def clear_stat(self):
        self.table_stat.reset_all()
        self.job_log.S = pd.DataFrame(columns=self.job_log.S.columns)
        for bm in self.bm_list:
            bm.clear()       

    def initial_events(self, main_model, stop_T=None, batch_dT=None):
        self.main_model = main_model
        if stop_T is not None:
            self.emq.cancelEvtMsgAll(main_model, "EM_stop")
            self.emq.scheduleEvtMsg4(stop_T, main_model, main_model, "EM_stop", param=True)
        if batch_dT is not None and batch_dT > 0:
            self.emq.cancelEvtMsgAll(main_model, "EM_batch")
            self.emq.scheduleEvtMsg4(batch_dT, main_model, main_model, "EM_batch", param=batch_dT)

    def run_simulation(self): 
        self.emq.runSimulation()
        return self.result

    def runSimulation(self): return self.run_simulation() # alias

    def register_pmodel(self, pmodel):  self.p_models.append(pmodel)

    def init_pmodel_process(self):
        for m in self.p_models:
            if m.need_to_start_proc:
                m.init_process()

    def register_bm(self, obj):  self.bm_list.append(obj)

    def save_batch_all(self):
        for bm in self.bm_list:
            bm.save_batch()

    def get_batch_df(self, fn=None):
        L = {bm.name: bm for bm in self.bm_list}
        df = pd.DataFrame(L)
        if fn is not None:
            with pd.ExcelWriter(fn, mode="w", engine='openpyxl') as writer:
                df.to_excel(excel_writer=writer)
        return df

    @staticmethod
    def set_seed(seed=42, all=False):
        if seed is None: return
        RndNumGen.init_rnd_generators(seed)
        # RndNumGen.set_master_seed(seed)  # PyDES의 master seed 고정
        if all:
            random.seed(seed)    # Python 내장 random 모듈
            np.random.seed(seed) # NumPy 모듈
            torch.manual_seed(seed) # PyTorch CPU 연산
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # PyTorch GPU 연산 (단일 및 다중 GPU)
            # # cuDNN 연산 결정론적 설정 (속도가 약간 감소할 수 있음)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    def run_with_mon(self, scene_fn, main_fn=None, ui=None, conn=True, time_scale=1):
        mon_proc = mp.Process(target=run_monitor, args=(ui, scene_fn, conn, self.config))
        mon_proc.start()
        self.emq.timeScale = time_scale
        self.emq.makeZmq(mon=True, sync=True)
        if main_fn is None:
            self.run_simulation()
        else:
            main_fn()
        mon_proc.join()

    def run_simulation_single(self, pid):
        torch.set_num_threads(1)
        Environment.current_env = self
        self.pid = pid
        # self.set_seed(pid)
        return self.run_simulation()

    def run_simulation_parallel(self, n_instance, pool=None, backend=None):
        if resolve(backend) == "jax":
            return run_parallel(self, n_instance)

        n_proc = min(n_instance, os.cpu_count())
        if pool is None:
            with mp.Pool(processes=n_proc) as p:
                return p.map(self.run_simulation_single, range(n_instance))
        else:
            return pool.map(self.run_simulation_single, range(n_instance))


    @staticmethod
    def run_multiple(work_fn, param_list, parallel=True, backend=None):
        if resolve(backend) == "jax":
            return run_multiple_jax(work_fn, param_list)

        if parallel:
            n_proc = min(len(param_list), os.cpu_count())
            # Pool is an old style multi-processing
            with mp.Pool(processes=n_proc) as pool:
                res_list = pool.starmap(work_fn, param_list)
            # ProcessPoolExecuter is a new style multi-processing
            # with ProcessPoolExecutor(max_workers=n_proc) as executor:
            #     futures = [executor.submit(work_fn, *p) for p in param_list]
            #     res_list = [f.result() for f in futures]                
        else:
            res_list = []
            for a in param_list:
                r = work_fn(*a)
                res_list.append(r)
        return res_list

    @staticmethod
    def collect_res_df(res_list, param_list=None, cfg_name_idx=None, col_name=None):
        res_dict = {}
        for i, df in enumerate(res_list):
            if cfg_name_idx is None:
                key = df.attrs["name"] 
            else: 
                key = param_list[i][cfg_name_idx]
            if col_name is None:
                res_dict[key] = df
            else:  # collect specified column only
                res_dict[key] = df[col_name]
        res_df = pd.DataFrame(res_dict)
        return res_df

    @staticmethod
    def make_plot(df, style="box"):
        if style == "box":
            df.boxplot()
        else:  # trend plot of mean
            df_mean = df.mean(axis=0)  # or use df.median
            plt.plot(df.columns, df_mean, color="blue", marker="o", label="Mean")
            # low = df.quantile(0.25, axis=0)
            # high = df.quantile(0.75, axis=0)
            std = df.std(axis=0) / np.sqrt(len(df))
            low = (df_mean - std).clip(lower=0)
            high = df_mean + std
            plt.fill_between(df.columns, low, high, color="blue", alpha=0.2, label="25th-75th Percentile")
        plt.show()

    @staticmethod
    def run_multiple_and_plot(work_fn, param_list, cfg_name_idx=None, col_name=None, style="box", parallel=True):
        res_list = Environment.run_multiple(work_fn, param_list, parallel)
        res_df = Environment.collect_res_df(res_list, param_list, cfg_name_idx, col_name)
        Environment.make_plot(res_df, style)
        return res_df

    def save_env(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self, f)    


    @staticmethod
    def load_env(fn):
        with open(fn, 'rb') as f:
            env = pickle.load(f)
            env.clear_stat()
        return env



class EvtMsg:
    def __init__(self, time, fm, to, msg, param=None):
        self.time = time
        self.fm = fm
        self.to = to
        self.msg = msg
        self.param = param
        self.canceled = False
        self.pause_time = 0.0

    def __lt__(self, other):
        return self.time < other.time

    def __str__(self):
        return f"{self.msg} @ {self.time:.3f}"

    def copy(self):  # e.canceled = False
        return EvtMsg(self.time, self.fm, self.to, self.msg, self.param)

# class PyDES_PQ has
# put(e) : enqueue
# get(e) : dequeue
# qsize() : # of items in queue
# empty() : check if empty

class EvtMsgQueue(PyDES_PQ):  #(PriorityQueue):
    def __init__(self, name, time_scale=0):
        super().__init__()
        self.name = name
        self.Tsince= 0.0
        self.Tnow = 0.0
        self.timeUnitConversion = 60  # sim clock unit is minute
        # timeScale: 0 for as fast as possible simulation, 1 for real time, 10 for 10x faster
        self.timeSince = time.time()
        self.timeScale = time_scale 
        self.timeStart = self.timeSince
        self.checkInterval = 4 * self.timeScale
        self.simZMQ = None
        self.monLog = None
        self.clockName = "simClock"
        self.paused_evt = []

    def clock(self):  return self.Tnow

    def reset_tnow(self, tnow_new=0.0):
        delta = tnow_new - self.Tnow
        self.Tnow = tnow_new
        for e in self.heap:  # maybe not in time order
            e.time += delta

    def scheduleEvtMsg4(self, dtime, fm, to, msg, param=None):
        e = EvtMsg(self.Tnow + dtime, fm, to, msg, param)
        self.put(e) # self.scheduleEvtMsg(e)

    def scheduleAgain(self, dtime, e):
        e = EvtMsg(self.Tnow + dtime, e.fm, e.to, e.msg, e.param)
        self.put(e) # self.scheduleEvtMsg(e)

    def findEvtMsg(self, to=None, msg=None, elist=None, param=None):
        if elist is None:
            elist = self.heap
        for i, e in enumerate(elist):  # not time order
            if self.match(e, to, msg, param):
                return e, i
        return None, None

    def cancelEvtMsg(self, to=None, msg=None, param=None):
        e, _ = self.findEvtMsg(to, msg, param=param)
        if e is not None:
            e.canceled = True
        return e
    
    def cancelEvtMsgAll(self, to=None, msg=None, pause=False, param=None):
        for e in self.heap:  # maybe not in time order
            if self.match(e, to, msg, param):
                e.canceled = True  # mark as canceled
                if pause:
                    pe = e.copy()  # pe.canceled = False
                    pe.time = e.time - self.Tnow
                    pe.pause_time = self.Tnow
                    self.paused_evt.append(pe)

    # delay = cancel + reschedule
    def delayEvtMsg(self, dt, to=None, msg=None):
        e = self.cancelEvtMsg(to, msg)
        if e:  # found
            de = e.copy() # e.canceled = False
            de.time += dt
            de.pause_time = self.Tnow
            self.scheduleEvtMsg(de)
        return e
    def delayEvtMsgAll(self, dt, to=None, msg=None, param=None):
        for e in self.heap:  # not time order
            if self.match(e, to, msg, param):
                e.canceled = True
                de = e.copy()  # e.canceled = False
                de.time += dt
                de.pause_time = self.Tnow
                self.scheduleEvtMsg(de)

    def pauseEvtMsg(self, to=None, msg=None):
        e = self.cancelEvtMsg(to, msg)
        if e:
            pe = e.copy() # pe.canceled = False
            pe.time = e.time - self.Tnow
            pe.pause_time = self.Tnow
            self.paused_evt.append(pe)
        return e
    def pauseEvtMsgAll(self, to=None, msg=None, param=None):
        self.cancelEvtMsgAll(to, msg, pause=True, param=param)

    def resumeEvtMsg(self, to=None, msg=None, param=None):
        e, i = self.findEvtMsg(to, msg, self.paused_evt, param)
        if e:
            self.paused_evt.pop(i)
            e.time += self.Tnow
            self.scheduleEvtMsg(e)
        return e
    def resumeEvtMsgAll(self, to=None, msg=None, msg_now=None, param=None):
        # need to process backward, so that index remains same after pop
        for i in range(len(self.paused_evt)-1,-1,-1):
            e = self.paused_evt[i]
            if self.match(e, to, msg, param):
                e = self.paused_evt.pop(i) 
                if msg_now and e.msg in msg_now:
                    dt = self.Tnow - e.pause_time
                    e.time -= dt
                    if e.time < 0.0:  e.time = 0.0
                e.time += self.Tnow
                self.scheduleEvtMsg(e)

    def clean(self):
        for i in range(self.len()-1,-1,-1):
            e = self.heap[i]
            if e.canceled:
                self.heap.pop(i)
        self.heapify()  # restore heap structure

    def scheduleEvtMsg(self, e):
        assert e.time >= self.Tnow, f"Scheduling past. Tnow={self.Tnow}, e.time={e.time}: {e}"
        # if e.time < self.Tnow:
        #     print (f"Scheduling event in the past.  Tnow={self.Tnow}, e.time={e.time}: {e}")
        self.put(e)
        # if self.timeScale <= 0:
        #     return 0
        # else:
        #     return (e.time - self.Tnow) / self.timeScale

    def peepTime(self):
        if self.empty(): return infinity
        return self.heap[0].time   #self.queue[0].time

    @staticmethod
    def isNoneOrEqual(a, b):
        return (a is None) or (a == b)

    @staticmethod
    def match(e, to, msg, param=None):
        return ((not e.canceled)
                and EvtMsgQueue.isNoneOrEqual(to, e.to)
                and EvtMsgQueue.isNoneOrEqual(msg, e.msg)
                and EvtMsgQueue.isNoneOrEqual(param, e.param)
                )

    def timeResync(self):
        self.Tsince = self.Tnow
        self.timeSince = time.time()

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state["simZMQ"]
    #     del state["monLog"]
    #     return state

    def ZMQ_unconnected(self):
        return self.simZMQ is None or (not self.simZMQ.connected)

    def ZMQ_connected(self):
        return self.simZMQ and self.simZMQ.connected

    def setMonLog(self, f):
        if self.monLog is not None:
            self.monLog.close()
        self.monLog = f

    # check message back from Monitor
    def checkMonMsg(self):
        if self.ZMQ_unconnected(): return
        msg = self.simZMQ.check_mon_msg()
        if msg is None: return
        print(msg)
        self.timeResync()
        m = msg.split()
        if m[0] == "timescale":
            self.timeScale = float(m[1])
            self.checkInterval = round(40*self.timeScale)
            if self.checkInterval <= 0: self.checkInterval = 1
        elif m[0] == "stop":  # stop sending messages
            self.simZMQ.setStopSending(True)
        elif m[0] == "restart": # restart sending
            self.simZMQ.setStopSending(False)

    def sleep_a_moment(self):
        if self.timeScale > 0:
            timeDelta = time.time() - self.timeSince  # wall clock
            dt = self.timeUnitConversion*(self.Tnow - self.Tsince) / self.timeScale - timeDelta
            if dt > 0.001:
                # if dt > 1:
                #     dt = 1
                #     self.timeResync()
                time.sleep(dt)
            elif dt < -1:
                self.timeResync()

    def advanceToNextEvt(self, ne):
        assert ne.time >= self.Tnow
        self.Tnow = ne.time
        self.sleep_a_moment()


    def sendMonMsg(self, msg):
        # print (msg)
        if self.ZMQ_unconnected():
            # if msg[:6] == 'widget':
            #     print(msg)
            return
        else:
            self.simZMQ.send_msg(msg)
        if self.monLog is not None:
            self.monLog.write(f"{self.Tnow:12.3f} {msg}\n")
        # print(msg)


    def runNextEvent(self):
        ne = self.get()
        if ne.canceled: return  # canceled event

        # print (self.name, "processes @", ne.time, ne.msg)
        Tprev = self.Tnow
        self.advanceToNextEvt(ne)
        if not ne.to.processEvtMsg(ne):
            print ("Event handler failed : ", ne.msg)
            exit(-1)

        if self.Tnow > Tprev:  # TODO : monitor crash check
            self.sendMonMsg(f"widget {self.clockName} text {self.Tnow:.3f}")
            self.checkMonMsg()

    def stop_running(self):
        self.keep_running = False

    def runSimulation(self):
        numEvt = 0
        # loop will stop if event handler sets self.keep_running = False
        self.keep_running = True 
        while self.keep_running and (not self.empty()):
            self.runNextEvent()
            numEvt += 1
            # if numEvt % 10000 == 0:
            #     print (f"# events processed = {numEvt} @ {self.Tnow:.2f}")
        # print (f"Total number of events processed = {numEvt} @ {self.Tnow:.2f}")


    def makeZmq(self, mon=False, sync=False):
        if self.ZMQ_connected(): return # already connected
        self.simZMQ = SimulationMQ()
        if mon:
            self.simZMQ.connect(sync=sync)
            time.sleep(0.5)  # wait a little
            self.timeResync()
        else:
            self.timeScale = 0


    def saveState(self, fileName):
        sys.setrecursionlimit(10**8)
        pickle.dump(self, fileName)
        # f = open(fileName, "wb")
        # monLog = self.monLog  # file handle can't be pickled
        # self.monLog = None
        # simZMQ = self.simZMQ
        # self.simZMQ = None
        # pickle.dump(self, f)
        #
        # self.monLog = monLog
        # self.simZMQ = simZMQ
        # f.close()

    def prepareQuit(self):
        pass


def loadState(fileName):
    f = open(fileName, "rb")
    emq = pickle.load(f)
    f.close()
    emq.timeResync()
    return emq


class EModel:
    def __init__(self, env=None):
        self.instantEvtMsg = self.directEvtMsg  # alias for the directEvtMsg function
        link_env(self, env)

    def Tnow(self): return self.emq.Tnow
    def env(self): return self.emq.env
    def table(self): return self.emq.env.table_stat

    def makeSceneObj(self, mo_book): # to be called by SimMon
        pass

    # directEvtMsg :process an event message immediately, without scheduling
    def directEvtMsg(self, to, msg, param=None):
        e = EvtMsg(None, self, to, msg, param) 
        to.processEvtMsg(e) 

    def directEvtMsg0(self, to, e):
        e.fm = self
        e.to = to
        to.processEvtMsg(e) 

    def scheduleEvtMsg4(self, dtime, fm, to, msg, param=None):
        self.emq.put(EvtMsg(self.emq.Tnow + dtime, fm, to, msg, param))
        # same as self.emq.scheduleEvtMsg4(dtime, fm, to, msg, param)

    def scheduleAgain(self, dtime, e):
        # same as self.emq.scheduleAgain(dtime, e)
        self.emq.put(EvtMsg(self.emq.Tnow + dtime, e.fm, e.to, e.msg, e.param))

    def processEvtMsg(self, e):
        # if e.msg != "EM_resume": print (e.time, e.msg)
        try:  # assume that EM_msg(e) methods are defined for each message
            handler = getattr(self, e.msg) # look for event handler method
        except AttributeError:
            print (f"Model {self}")
            print (f"EvtMsg handler is not defined : {e.msg}")
            print (f"from {e.fm} to {e.to}")
            exit(-1)
            # return False

        handler(e)
        return True

    def EM_batch(self, e):
        # print (f"batch saved @ {self.Tnow():.3f}")
        self.env.save_batch_all()
        batch_dT = e.param
        self.scheduleAgain(batch_dT, e)

    def EM_stop(self, e):
        # self.emq.cancelEvtMsgAll()
        self.emq.stop_running()


class PModel(EModel):
    def __init__(self, env=None):
        super().__init__(env)
        self.env.register_pmodel(self)
        self.need_to_start_proc = True
        self.process_instance = None
        self.waiting_for = None
        self.timeout_evt = False
        self.EM_next = self.EM_resume
        self.activity_cycle = self.process_cycle  # alias for process_cycle function
        self.resume_next = self.resume_process # alias
        self.schedule_next = self.schedule_resume  # alias
        self.start_process = self.init_process # alias

    # for pickling purpose, generator instances should be removed
    def __getstate__(self):
        state = self.__dict__.copy()
        try:
            del state["_proc_instance_"]
        except:
            pass
        try:
            del state["emq"]
        except:
            pass
        state["need_to_start_proc"] = True
        return state

    def __setstate__(self, state):
        # self.__dict__ = state
        self.__dict__.update(state)
        PModel.register(self)
        self.need_to_start_proc = True

    def process_cycle(self):
        pass

    def init_process(self, proc_func=None, proc_name=None):
        if self.emq is None:
            return
        # create process generator
        if proc_func is None:
            self.process_instance = self.process_cycle()
        else:
            self.process_instance = proc_func()
        # register process to self
        if proc_name is None:
            proc_name = "_proc_instance_"
        self.__dict__[proc_name] = self.process_instance
        # start process
        self.need_to_start_proc = False
        next(self.process_instance)


    def stop_process(self):
        if self.process_instance is None:
            print (f"No process instance to stop")
            return
        self.emq.cancelEvtMsg(to=self, msg="EM_resume", param=None) # cancel timeout event
        self.emq.cancelEvtMsg(to=self, msg="EM_timeout", param=None) # cancel timeout event
        self.waiting_for = None
        self.process_instance = None
        

    def schedule_resume(self, dt, proc_instance=None, param=None, msg="EM_resume"):
        self.wait_event(msg)
        if proc_instance is None:
            self.emq.scheduleEvtMsg4(dt, self, self, msg, param=param)
        elif param is None:  # proc is given, param is None
            self.emq.scheduleEvtMsg4(dt, self, self, msg, param=proc_instance)
        else:  # proc is given, param is given
            self.emq.scheduleEvtMsg4(dt, self, self, msg, param=(proc_instance, param))


    def wait_event(self, evt, timeout=None):
        self.waiting_for = evt
        if timeout is not None:
            self.emq.scheduleEvtMsg4(timeout, self, self, "EM_timeout")
            self.timeout_evt = True


    def resume_process(self, proc_instance=None, ret_val="default", check_evt=None):
        assert check_evt == self.waiting_for, f"process is waiting for {self.waiting_for}, but get {check_evt}"
        try:
            if proc_instance is None:
                proc_instance = self._proc_instance_
            if self.timeout_evt:
                self.emq.cancelEvtMsg(to=self, msg="EM_timeout", param=None) # cancel timeout event
                self.timeout_evt = False
            proc_instance.send(ret_val)
        except StopIteration as exc:
            pass
        except ValueError as exc:
            print ("resume_process, value error:", exc)
            print (f"self = {self}")

    # def notify_resource(self, res):
    #     self.resume_process(self._proc_instance_, res)

    # timeout is a kind of EM_resume
    def EM_timeout(self, e=None):
        self.waiting_for = "EM_resume"
        self.EM_resume(e)

    def EM_resume(self, e=None):
        evt = "EM_resume"
        if e is None or e.param is None:
            self.resume_process(self._proc_instance_, None, check_evt=evt)
        elif inspect.isgenerator(e.param):  # assume e.param is the process to resume
            self.resume_process(e.param, None, check_evt=evt)
        elif isinstance(e.param, tuple): # e.param = (process, param)
            self.resume_process(e.param[0], e.param[1], check_evt=evt)
        else:  # assume e.param is the process to resume
            self.resume_process(self._proc_instance_, e.param, check_evt=evt)



# a queue waiting for resource
class ResourceQueue(list):
    def __init__(self, id=None, log_level=0, env=None):
        self.id = id
        self.log_level = log_level  # 0 for no log, 1 for average only, 2 for detailed log
        if log_level > 0:
            self.log = TimeAverage(name=id, v=0, history=log_level>1)
        link_env(env)

    def Tnow(self):
        return self.emq.Tnow

    def empty(self):
        return len(self) == 0

    def enqueue(self, job):
        self.append(job)
        if self.log_level > 0:
            self.log.stateInc(1)


    def dequeue(self, i=0):
        if len(self) > 0:
            job = self.pop(i)
            if self.log_level > 0:
                self.log.stateInc(-1)
            return job
        else:
            return None

    def remove(self, job):
        try:
            i = self.index(job)
            self.dequeue(i)
        except Exception as e:
            pass
            # print(f"job {job} is not found in the queue {self.id}")

    def peep(self):
        return self[0]


class Resource:  # resource with capacity
    def __init__(self, id=None, capa=1, log_level=0, env=None, RQ=None):
        self.id = id
        self.capa = capa  # initially available
        self.in_use = 0
        self.seized_by = []
        self.log_level = log_level # 0 for no log, 1 for average only, 2 for detailed log
        link_env(env)
        if RQ is not None:
            self.queue = RQ(id, log_level, env)  # queue of entities waiting for this resource
        if self.log_level > 0:
            self.log = TimeAverage(name=id, v=self.capa, history=log_level>1)

    def init_state(self, state_var_name="state", s=None):
        self.__dict__[state_var_name] = s
        # to be used as self.var1 if sv_name=='var1'

    def Tnow(self):
        return self.emq.Tnow

    def available(self):
        return self.capa

    def unavailable(self):
        return self.capa == 0

    def update_log(self):
        if self.log_level > 0:
            self.log.stateChange(self.available())

    def is_seized_by(self, job):
        return job in self.seized_by

    def is_seized_or_queued(self, job):
        return self.is_seized_by(job) or job in self.queue

    def set_seized_by(self, job):
        self.seized_by.append(job)

    def reset_seized_by(self, job):
        self.seized_by.remove(job)

    def get(self):  # get resource
        self.capa -= 1
        self.in_use += 1
        self.update_log()
        return self

    def put(self, r=None):  # return resource
        self.capa += 1
        self.in_use -= 1
        self.update_log()

    def seize(self, job):
        if self.available():
            r = self.get()
            r.set_seized_by(job)
            if job and job.waiting_for == self:
                job.waiting_for = None
        else:
            if job:  # queue if job is not None
                self.queue.enqueue(job)
                job.waiting_for = self
            r = None
        return r

    def release(self, job, r=None):
        self.put(r)
        if r is None:
            self.reset_seized_by(job)
        else:
            r.reset_seized_by(job)
        if job:
            job.waiting_for = None

    def dequeue(self, seize=True):
        if (not self.available()) or len(self.queue) == 0:
            return None, None
        job = self.queue.dequeue()
        if seize:
            r = self.seize(job)
        else:
            r = None
        return job, r

    def cancel_waiting(self, job):
        if job in self.queue:
            self.queue.remove(job)

    def release_dequeue_callback(self, job, r=None, callback=None, seize=True):
        self.release(job, r)
        job, r = self.dequeue(seize)
        if job is None: return
        if callback == False:
            return
        if (callback is None) and (job._proc_instance_ is not None):  # None or False
            job.resume_process(ret_val=r)
        elif inspect.isgenerator(callback):
            job.resume_process(proc_instance=callback, ret_val=r)
        elif callable(callback):
            callback(job)  # calls directly


class Resource1(Resource):  # resource with capacity = 1 (this can be seized only once)
    def __init__(self, id=None, log_level=0, env=None, RQ=None):
        super().__init__(id=id, capa=1, log_level=log_level, env=env, RQ=RQ)
        self.seized_by = None

    def is_seized_by(self, job):
        return job == self.seized_by

    def set_seized_by(self, job):
        self.seized_by = job

    def reset_seized_by(self, job):
        self.seized_by = None

    def release(self, job, r=None):
        assert r is None or r is self
        self.put(r)
        self.seized_by = None
        if job:
            job.waiting_for = None



class ResourceSet(Resource):  # set of individual resources
    def __init__(self, id, capa=1, log_level=0, env=None, RT=Resource1, RQ=ResourceQueue):
        super().__init__(id, capa, log_level, env, RQ)
        self.pool = [RT(id=f"{id}_{i}") for i in range(capa)]

    def init_state(self, sv_name="state", s=None):
        for r in self.pool:
            r.init_state(sv_name, s)

    def available(self):
        return len(self.pool)

    def get(self):
        r = self.pool.pop()
        self.in_use += 1
        self.update_log()
        return r

    def put(self, r):
        self.pool.append(r)
        self.in_use -= 1
        self.update_log()




class JobSource(PModel):
    def __init__(self, env, job_class, to_model, iat_gen, msg="EM_job_arrival"):
        super().__init__(env)
        self.job_class = job_class
        self.to_model = to_model
        self.msg = msg
        self.count = 0
        self.iat_gen = iat_gen

    def __getstate__(self):
        state = super().__getstate__()
        try:
            del state["iat_generator"]
        except:
            pass
        return state

    def make_job(self):
        if self.job_class:
            job = self.job_class(self.count)
            self.count += 1
        else:
            job = None
        return job

    def process_cycle(self):
        while True:
            dt = self.iat_gen.get()
            if dt is None:
                break
            yield self.schedule_resume(dt)
            job = self.make_job()
            self.directEvtMsg(self.to_model, self.msg, param=job)

    def get_count(self):
        return self.count


# alias (ond names)
MyQueue = PyDES_Queue
MyPQ = PyDES_PQ
AModel = EModel     # old name for EModel
ACModel = PModel    # old name for PModel

    


# # pickle serialization test for circular reference
# class A:
#     def __init__(self):
#         self.name = 'A'
#         self.num = 100
#         self.mate = None
#     def pr(self):
#         print (self.name, self.num, self.mate.name)
#
# class B:
#     def __init__(self):
#         self.name = 'B'
#         self.num = 200
#         self.mate = None
#
#     def pr(self):
#         print (self.name, self.num, self.mate.name)
#
# if __name__ == "__main__":
#     a = A()
#     b = B()
#     a.mate = b
#     b.mate = a
#     a.pr()
#     b.pr()
#
#     s = pickle.dumps(a)
#     print(s)
#     c = pickle.loads(s)
#     print(c)
#     c.pr()

