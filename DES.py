# from queue import *
import time
from ConnectMQ import *
import math
from Utility import *
import pickle
import sys
import inspect
from itertools import count


# single federation version for simplicity

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

# class MyPQ has
# put(e) : enqueue
# get(e) : dequeue
# qsize() : # of items in queue
# empty() : check if empty

class EvtMsgQueue(MyPQ):  #(PriorityQueue):
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

    def scheduleEvtMsg4(self, dtime, fm, to, msg, param=None):
        e = EvtMsg(self.Tnow + dtime, fm, to, msg, param)
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
        if e:
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["simZMQ"]
        del state["monLog"]
        return state

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

    def runSimulation(self, t_end=None, main_model=None):
        if t_end is not None and main_model is not None:
            self.scheduleEvtMsg4(t_end, None, main_model, "EM_stop")
        numEvt = 0
        while not self.empty():
            self.runNextEvent()
            numEvt += 1
            # if numEvt % 10000 == 0:
            #     print (f"# events processed = {numEvt} @ {self.Tnow:.2f}")
        print (f"Total number of events processed = {numEvt} @ {self.Tnow:.2f}")


    def makeZmq(self, mon=False, sync=False):
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
    def __init__(self, emq=None):
        self.instantEvtMsg = self.directEvtMsg  # alias for the directEvtMsg function
        self.set_emq(emq)

    def Tnow(self):
        return self.emq.Tnow

    def set_emq(self, emq):
        self.emq = emq

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
        self.emq.scheduleEvtMsg4(dtime, fm, to, msg, param)
        # same as self.emq.put(EvtMsg(dtime, fm, to, msg, param))

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
        BatchMean.save_batch_all(self.Tnow())
        batch_dT = e.param
        self.scheduleEvtMsg4(batch_dT, self, self, "EM_batch", param=batch_dT)

    def EM_stop(self, e):
        self.emq.cancelEvtMsgAll()


class PModel(EModel):
    p_models = []

    @classmethod
    def register(cls, model):
        cls.p_models.append(model)

    @classmethod
    def init_process_all(cls):
        for m in cls.p_models:
            if m.need_to_start_proc:
                m.init_process()


    def __init__(self, emq=None):
        super().__init__(emq)
        PModel.register(self)
        self.need_to_start_proc = True
        self.activity_cycle = self.process_cycle  # alias for process_cycle function
        self.waiting_for = None
        self.timeout_evt = False
        self.resume_next = self.resume_process # alias
        self.EM_next = self.EM_resume
        self.schedule_next = self.schedule_resume

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
            process_instance = self.process_cycle()
        else:
            process_instance = proc_func()
        # register process to self
        if proc_name is None:
            proc_name = "_proc_instance_"
        self.__dict__[proc_name] = process_instance
        # start process
        self.need_to_start_proc = False
        next(process_instance)


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


AModel = EModel     # old name for EModel
ACModel = PModel    # old name for PModel


# a queue waiting for resource
class ResourceQueue(list):
    def __init__(self, id=None, log_level=0, emq=None):
        self.id = id
        self.log_level = log_level  # 0 for no log, 1 for average only, 2 for detailed log
        if log_level > 0:
            self.log = TimeAverage(t=emq.Tnow, v=0, history=log_level>1)
        self.emq = emq

    def Tnow(self):
        return self.emq.Tnow

    def empty(self):
        return len(self) == 0

    def enqueue(self, job):
        self.append(job)
        if self.log_level > 0:
            self.log.stateInc(self.Tnow(), 1)


    def dequeue(self, i=0):
        if len(self) > 0:
            job = self.pop(i)
            if self.log_level > 0:
                self.log.stateInc(self.Tnow(), -1)
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
    def __init__(self, id=None, capa=1, log_level=0, emq=None, RQ=None):
        self.id = id
        self.capa = capa  # initially available
        self.in_use = 0
        self.seized_by = []
        self.log_level = log_level # 0 for no log, 1 for average only, 2 for detailed log
        self.emq = emq  # emq is required if log_lvel > 0
        if RQ is not None:
            self.queue = RQ(id, log_level, emq)  # queue of entities waiting for this resource
        if self.log_level > 0:
            self.log = TimeAverage(t=0, v=self.capa, history=log_level>1)

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
            self.log.stateChange(self.Tnow(), self.available())

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
    def __init__(self, id=None, log_level=0, emq=None, RQ=None):
        super().__init__(id=id, capa=1, log_level=log_level, emq=emq, RQ=RQ)
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
    def __init__(self, id, capa=1, log_level=0, emq=None, RT=Resource1, RQ=ResourceQueue):
        super().__init__(id, capa, log_level, emq, RQ)
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
    def __init__(self, emq, job_class, to_model, iat_gen, msg="EM_job_arrival"):
        super().__init__(emq)
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

