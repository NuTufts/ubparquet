import os,sys

class DetectorOutline:
    def __init__(self):
        self.tpc = [[0.0,256.0],
                    [-117.0,117.0],
                    [0.0,1036.0]]
        self.dettick_range = [0.0, 9600.0]
        self.tpctrig_tick = 3200.0
        self.detx_range = [ (self.dettick_range[0]-self.tpctrig_tick)*0.5*0.111,
                            (self.dettick_range[1]-self.tpctrig_tick)*0.5*0.111 ]

        self.top_pts  = [ [self.tpc[0][0],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][0]] ]
        self.bot_pts  = [ [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]] ]
        self.up_pts   = [ [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][0]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][0]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][0]] ]
        self.down_pts = [ [self.tpc[0][0],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][1],self.tpc[1][0], self.tpc[2][1]],
                          [self.tpc[0][1],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][1], self.tpc[2][1]],
                          [self.tpc[0][0],self.tpc[1][0], self.tpc[2][1]] ]
                
    def getlines(self,color=(255,255,255)):

        # top boundary
        Xe = []
        Ye = []
        Ze = []

        for boundary in [self.top_pts, self.bot_pts, self.up_pts, self.down_pts]:
            for ipt, pt in enumerate(boundary):
                Xe.append( pt[0] )
                Ye.append( pt[1] )
                Ze.append( pt[2] )
            Xe.append(None)
            Ye.append(None)
            Ze.append(None)
        
        
        # define the lines to be plotted
        lines = {
            "type": "scatter3d",
            "x": Xe,
            "y": Ye,
            "z": Ze,
            "mode": "lines",
            "name": "",
            "line": {"color": "rgb(%d,%d,%d)"%color, "width": 5},
        }
        
        return [lines]

                
