# from DES import *
from Utility import *
from heapq import heappush, heappop, heapify, _siftdown
# import pandas as pd

def heapup(heap, obj):
    i = heap.index(obj)
    _siftdown(heap, 0, i)

class Vertex:
    def __init__(self, name, x=0, y=0, type=None):
        self.name = name
        self.x = x
        self.y = y
        self.type = type
        self.outEdges = []
        self.inEdges = []

    def __lt__(self, other):
        return self.dist < other.dist

    def __str__(self):
        # return f"Vtx{self.name}({self.type})"
        # return f"Vtx{self.name}"
        return self.name

    def is_port(self):
        return self.type in ["U", "L", "LU", "BC", "W", "F", "E", "B"]

    def no_way_in(self):
        return len(self.inEdges) == 0

    def no_way_out(self):
        return len(self.outEdges) == 0


    def is_dead_end(self):
        if len(self.outEdges) != 1 or len(self.inEdges) != 1:
            return False
        return self.inEdges[0].pair == self.outEdges[0]

    # box = ((x0,-y0), (x1,-y1))
    def is_in_box(self, box):
        return box[0][0] < self.x < box[1][0] and -box[1][1] < self.y < -box[0][1]

    def get_box_name(self, boxes):
        for bn, box in boxes.items():
            if self.is_in_box(box):
                return bn
        # return None
        return "bu"

    def has_single_in_edge(self):
        for e in self.inEdges:
            if e.pair is None:
                return True
        return False


    def getPos(self):
        return self.x, self.y

    def findOutEdge(self, ev):
        for e in self.outEdges:
            if e.ev == ev:
                return e
        return None

    def findInEdge(self, sv):
        for e in self.inEdges:
            if e.sv == sv:
                return e
        return None

    def diff(self, ev):
        dx = ev.x - self.x
        dy = ev.y - self.y
        return np.array([dx, dy])

    def distTo(self, ev):
        return np.linalg.norm(self.diff(ev))

    def stepTo(self, ev, t=0.0):
        if t == 0.0:
            return self.getPos()
        t1 = 1 - t
        x = t1*self.x + t*ev.x
        y = t1*self.y + t*ev.y
        return x, y


class Edge:
    def __init__(self, sv, ev, cost=None, type=None):
        self.sv = sv
        self.ev = ev
        self.type = type
        self.pair = None
        sv.outEdges.append(self)
        ev.inEdges.append(self)

        self.setAngle()
        if cost is None:
            cost = self.length
        self.tt = cost  # travel time

    def id(self):
        # return self.sv.name+'-'+self.ev.name
        return f"E{self.sn}"

    def __str__(self):
        return f"~{self.ev}:{self.type}"

    def getPos(self, t=0.0):
        return self.sv.stepTo(self.ev)

    def getMid(self):
        return self.getPos(0.5)

    def setAngle(self):
        self.dv1 = self.sv.diff(self.ev)
        self.length = np.linalg.norm(self.dv1)
        if self.length == 0.0:
            self.dv1 = np.array([1.,0.])
        else:
            self.dv1 /= self.length
        theta = math.atan2(self.dv1[1], self.dv1[0])
        self.angle = int(math.degrees(theta))
        self.dv2 = np.array([-self.dv1[1], self.dv1[0]])

    def get_uv(self, vtx):
        d = self.sv.diff(vtx)
        u = np.dot(d, self.dv1)
        v = np.dot(d, self.dv2)
        return u, v

    def pre_pickle(self):
        self.sv = self.sv.name
        self.ev = self.ev.name

    def post_pickle(self, Vmap):
        self.sv = Vmap[self.sv]
        self.ev = Vmap[self.ev]

    def remove(self, Elist):
        try:
            self.sv.outEdges.remove(self)
        except:
            pass
        try:
            self.ev.inEdges.remove(self)
        except:
            pass
        try:
            Elist.remove(self)
        except:
            pass


class Graph:
    def __init__(self, name, use_adv_class=False):
        self.name = "Graph"
        self.fn = name
        self.Vmap = {}
        self.Elist = []
        self.VertexClass = Vertex
        self.EdgeClass = Edge

    def find_edge(self, sv, ev):
        for e in sv.outEdges:
            if e.ev == ev:
                return e
        assert False, "No edge found"
        return None

    def makeVertex(self, name="", x=0, y=0, type=None):
        v = self.VertexClass(name, x, y, type)
        self.Vmap[name] = v
        return v

    @staticmethod
    def modifyPos(type, sv, ev, mv, i):
        if type is None or type[0] == 'L':
            return
        dx = ev.x - sv.x
        dy = ev.y - sv.y
        d = math.sqrt(dx*dx + dy*dy)
        if type[0] == 'D': #'C':
            d /= 4
        elif type[0] == 'C': #'D':
            d /= -4
        else:  # type[0] == 'S'
            d /= 10
            sign = (dx * dy) * (i - 0.5)
            # if sign > 0:
            if sign < 0:
                d = -d
        mx, my = 0, -d
        mv.x, mv.y = rotate(dx, dy, mx, my, mv.x, mv.y)

    def checkPair(self, e):
        for pe in self.Elist:
            if e.sv == pe.ev and e.ev == pe.sv:
                e.pair = pe
                pe.pair = e
                return True
        return False

    def addEdge(self, sv, ev, c=None, type=None, nsplit=0):
        vlist = [sv]
        if nsplit > 0:
            for i in range(nsplit):
                t = (i+1)/(nsplit+1)
                x,y = sv.stepTo(ev, t)
                tv = Vertex("", x, y) # temporary midpoint
                self.modifyPos(type, sv, ev, tv, i)  # internal point position
                mv = self.getVertexXY(tv.x, tv.y, 'I', tol=1, new=True)
                vlist.append(mv)
        vlist.append(ev)

        n = len(vlist)
        for i in range(1, n):
            av = vlist[i-1]
            bv = vlist[i]
            if c is not None:  # divide edge cost
                c /= (n-1)
            e = self.EdgeClass(av, bv, c, type)
            e.sn = len(self.Elist)  # serial number
            self.Elist.append(e)
            self.checkPair(e)
        return e

    def getVertex(self, name):
        try:
            v = self.Vmap[name]
        except KeyError:
            print (f"vertex {name} is not defined")
            v = self.makeVertex(name)
        return v

    @staticmethod
    def round(v, tol=50):
        return np.round(v / tol) * tol

    def closestVertex(self, x, y):
        vtx = min(self.Vmap.values(),
                      key=lambda v: abs(v[0] - x) + abs(v[1] - y)
                )
        return vtx

    # Vmap에서 (x,y) 찾아서 vertex id를 return (없으면 등록)
    def getVertexXY(self, x, y, type='P', tol=50, new=True):
        rx = self.round(x, tol)
        ry = self.round(y, tol)
        vid = f"V({rx:.0f},{-ry:.0f})"
        try:
            v = self.Vmap[vid]
        except KeyError:
            if new:
                v = self.makeVertex(vid, rx, ry, type)
            else:
                v = self.closestVertex(x, y)
                # d = distance from (x,y) to v
                # if d > tol * 2:
                #     print(f"{vid} does not exist at ({x:.0f}, {y:.0f})")
                #     print(f"closest dist = {d}, v_min = {v_min}")
        return v

    # check for isolated vertices
    def check_isolated_vertex(self):
        nucv, nin, nout = 0, 0, 0
        isolated_vertex = []
        for id, v in self.Vmap.items():
            no_in = len(v.inEdges) == 0
            no_out = len(v.outEdges) == 0
            if no_in: nin += 1
            if no_out: nout += 1
            if no_in or no_out:
                isolated_vertex.append(id)
                nucv += 1
        print (f"No-in = {nin}, No-out = {nout}, isolated = {nucv} removed")
        for id in isolated_vertex:
            v = self.Vmap.pop(id)  # remove unconnected vertices
            for e in v.inEdges + v.outEdges:
                e.remove(self.Elist)


    def readGraph(self, vFileName=None, eFileName=None, split=0, ppv=None, ppe=None, scale=1):
        print ("Start reading graph")
        if vFileName is None:
            vFileName = f"{self.fn}_V.txt"
        vFile = open(vFileName, 'r')
        for line in vFile:
            tk = line.split()
            if len(tk) == 0: continue
            if tk[0][0] == "#": continue  # comment line
            type = tk[3] if len(tk) > 3 else None
            v = self.makeVertex(tk[0], scale*float(tk[1]), -scale*float(tk[2]), type)
            if ppv is not None:
                ppv(v, tk)
        vFile.close()

        if eFileName is None:
            eFileName = f"{self.fn}_E.txt"
        eFile = open(eFileName, 'r')
        for line in eFile:
            tk = line.split()
            if len(tk) == 0: continue
            if tk[0][0] == "#": continue  # comment line
            sv = self.getVertex(tk[0])
            ev = self.getVertex(tk[1])
            if len(tk) > 2:
                if tk[2].isdecimal():
                    c = float(tk[2])
                    type = None
                else:
                    c = None
                    type = tk[2]
            if type[0] != 'L':  # curved edges (C, D, S type)
                nsplit = 1
            elif split > 0:
                nsplit = int(tk[-1])
                if nsplit < 0:
                    d = sv.distTo(ev)
                    nsplit = int(np.floor(d/(split*scale)))
            else:
                nsplit = 0
            self.addEdge(sv, ev, c, type, nsplit=nsplit)
        eFile.close()

        self.check_isolated_vertex()
        print ("End reading graph")




    def pre_pickle(self):
        for e in self.Elist:
            e.pre_pickle()


    def post_pickle(self):
        for e in self.Elist:
            e.post_pickle(self.Vmap)

    @staticmethod
    def checkProxi(v, plist, tol=500):
        for p in plist:
            d = abs(v.x-p[0]) + abs(v.y-p[1])
            if d < tol: return True
        return False



    def reset(self):
        for v in self.Vmap.values():
            v.dist = infinity
            v.flag = False
            v.prev = None

    def sanityCheck(self, sv, dv, route):
        assert sv == route[-1]
        assert dv == route[0]
        pv = dv
        # each vertex should appear only once
        self.reset()
        for v in route:
            assert v.flag == False
            v.flag = True
            assert pv.distTo(v) < 100
            pv = v

    def vlist2elist(self, vlist):
        n = len(vlist)
        elist = []
        for i in range(n-1,0,-1):
            sv = vlist[i]
            ev = vlist[i-1]
            e = sv.findOutEdge(ev)
            if e is not None:
                elist.append(e)
        return elist

    def backtrack(self, sv, dv, route):
        # sv, dv: source & destination Vertex
        if dv is None:
            return False
        route.append(dv)
        if sv == dv:
            return True
        return self.backtrack(sv, dv.prev, route)

    def get_route_time(self, route, nominal=True):
        rt = 0.0
        for e in route:
            if nominal:
                rt += e.nominal_tt
            else:
                rt += e.tt
        return rt


    # def Dijkstra(self, sv, dv=None, emq=None):   # emq for displaying edge connection
    def Dijkstra(self, sv, dv=None):  #, emq=None):
        # sName, eName: string (name of source & destination Vertex)
        # sv, ev = graph.getVertex(sName), graph.getVertex(dName)
        # print (sv.name)
        self.reset()
        sv.dist = 0
        heap = []
        heappush(heap, (sv.dist,sv))
        while len(heap) > 0:
            v = heappop(heap)[1]      # v = pq.get()
            if v is dv: break  # destination reached
            if v.flag: continue  # v is already processed
            v.flag = True
            # print (v.dist)

            for e in v.outEdges:
                if e.blocked: continue
                # if emq:  # TODO : delete
                #     GEi = f"{self.fn}.E{e.sn}"
                #     emq.sendMonMsg(f"color {GEi} 9")
                w = e.ev
                if w.flag: continue
                temp = v.dist + e.tt
                if w.dist > temp:
                    w.dist = temp
                    w.prev = v
                    heappush(heap, (w.dist, w))

        if dv is not None:  # find explicit route (edge list)
            vlist = []
            if self.backtrack(sv, dv, vlist): # reversed sequence of vertices
                route = self.vlist2elist(vlist)
                return route
        return None



    # def Dijkstra(self, sv, dv=None):
    #     # sName, eName: string (name of source & destination Vertex)
    #     # sv, ev = graph.getVertex(sName), graph.getVertex(dName)
    #     # print (sv.name)
    #     self.reset()
    #     sv.dist = 0
    #     # pq = MyPQ()  # PriorityQueue()
    #     heap = []
    #     # pq.put((sv.dist,sv))  # pq.put(sv) 안됨 heap의 update가 안되기에...
    #     heappush(heap, sv)
    #     # while not pq.empty():
    #     while len(heap) > 0:
    #         # v = pq.get()[1]      # v = pq.get()
    #         v = heappop(heap)
    #         if v is dv: break  # destination reached
    #         if v.flag: continue  # v is already processed
    #         v.flag = True
    #         # print (v.dist)
    #
    #         for e in v.outEdges:
    #             if e.blocked: continue
    #             w = e.ev
    #             if w.flag: continue
    #             # temp = v.dist + e.adjusted_tt(dv)
    #             dist_via_v = v.dist + e.tt
    #             if dist_via_v < w.dist:
    #                 new = (w.dist == infinity)
    #                 w.dist = dist_via_v
    #                 w.prev = v
    #                 if new:
    #                     heappush(heap, w)
    #                 else:
    #                     heapup(heap, w)
    #
    #     if dv is not None:  # find explicit route (edge list)
    #         vlist = []
    #         if self.backtrack(sv, dv, vlist): # reversed sequence of vertices
    #             route = self.vlist2elist(vlist)
    #             return route
    #     return None

    def getDistMap(self, sv):
        self.Dijkstra(sv)
        dmap = {}
        for ev in self.Vmap.values():
            dmap[ev.name] = ev.dist
            if ev.dist == infinity:
                print(f"No path between {sv.name} and {ev.name}")
        return dmap

    def DijkstraAllPair(self):
        print ("Start: Computing reference distance map")
        self.Dmat = {}
        for sv in self.Vmap.values():
            # self.Dijkstra(sv) --> getDistMap에서 호출
            dm = self.getDistMap(sv)
            self.Dmat[sv.name] = dm
        print ("End: Computing reference distance map")

    # # Bellman-Ford alg. using v.dist and v.prev
    # def BellmanFord(self, sv):
    #     self.reset()
    #     sv.dist = 0
    #     for k in range(len(self.Vmap.values())):
    #         for e in self.Elist:
    #             newDist = e.sv.dist + e.cost
    #             if newDist < e.ev.dist:
    #                 e.ev.dist = newDist
    #                 e.ev.prev = e.sv
    #         # print("k = ", k, "dist = ", vt.dist)
    #
    #     # check negative cycle
    #     for e in self.Elist:
    #         if e.sv.dist + e.cost < e.ev.dist:
    #             print("Negative cycle found.")
    #             return False
    #     # backtrack : use Backtrack(sv, ev) function above
    #     return True


    # # for a sparse graph, repeated use of Dijkstra is more efficient than FLoyd-Warshall
    # def FloydWarshall (self):
    #     NV = len(self.Vmap.values())
    #     NVxNV = (NV, NV)
    #     rNV = range(NV)
    #     dist = np.full(NVxNV, infinite)
    #     prev = np.full(NVxNV, -1, dtype='int')
    #
    #     # dist from vertex to itself is set to 0
    #     # initialize the path matrix
    #     for i in rNV:
    #         dist[i][i] = 0
    #         prev[i][i] = i
    #
    #     for e in self.Elist :
    #         i = e.sv.id
    #         j = e.ev.id
    #         if i >= NV or j >= NV: continue
    #         dist[i][j] = e.cost
    #         prev[i][j] = i
    #
    #     # Floyd-Warshall main loop
    #     for k in rNV:
    #         # print('k = ', k)
    #         for i in rNV:
    #             if dist[i][k] == infinite: continue
    #             for j in rNV:
    #                 if dist[k][j] == infinite: continue
    #                 newdist = dist[i][k] + dist[k][j]
    #                 if newdist < dist[i][j]:
    #                     dist[i][j] = newdist
    #                     prev[i][j] = prev[k][j]
    #
    #     # display shortest paths
    #     def backtrackFW(i, j):
    #         if dist[i][j] == infinite:
    #             return " no path to "
    #         if prev[i][j] == i:
    #             return " "
    #         else:
    #             pij = prev[i][j]
    #             pv = self.V[pij]
    #             return backtrackFW(i, pij) + pv.name + backtrackFW(pij, j)


    def print(self):
        print ("Vertex name map")
        for k, v in self.Vmap.items():
            print (k, v.name)

        print ("Edge list")
        for e in self.Elist:
            print (e.sv.name, e.ev.name, e.tt)

    def set_attrib(self):
        # for v in self.Vmap.values():
        #     v.set_attrib()
        for e in self.Elist:  # G_config에 있는 값 적용
            e.set_attrib()


def read_network(name, GraphClass, split=0, Dmat=True, ppv=None, ppe=None, scale=1):
    pkl_fn = f"{name}_graph.pkl"
    src_fn = f"{name}_V.txt"
    g = pickle_load_graph(pkl_fn, src_fn)
    # g = None
    if g is None:  # no pickle file
        g = GraphClass(name)
        g.readGraph(split=split, ppv=ppv, ppe=ppe, scale=scale)
        pickle_dump_graph(g, pkl_fn)
    g.set_attrib()  # set config related attributes of an edge
    # p = name.rfind("/")
    # if p > 0:
    #     g.name = name[p+1:]


    print("Reading network done")
    if Dmat:
        dmat_pkl_fn = f"{name}_Dmat.pkl"
        g.Dmat = pickle_load(dmat_pkl_fn, src_fn)
        # g.Dmat = None
        if g.Dmat is None:
            g.DijkstraAllPair()
            pickle_dump(g.Dmat, dmat_pkl_fn)
    return g


if __name__ == "__main__":
    g = Graph(use_adv_class=True)
    cn = "AGV_24bay"
    g.readGraph(f'{cn}_V.txt', f'{cn}_E.txt')
    g.DijkstraAllPair()
    Vlist = g.Vmap.values()
    for sv in Vlist:
        for ev in Vlist:
            print (sv.name, ev.name, g.Dmat[sv][ev])
    # print(g.Vmap)