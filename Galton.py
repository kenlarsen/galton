'''
Galton board simulator
    
Kenneth Larsen
kenbolarsen@gmail.com

'''

import sys, math, random, time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import logging
from scipy.interpolate import interp1d
from scipy.stats import norm

logging.basicConfig(level=logging.INFO)

GFORCE          = 9.8 * 15 # gravity
BALL_COUNT      = 30  # # of simultaneous balls in play
BALL_RADIUS     = 6   # size of the balls
PEG_RADIUS      = 4   # size of the pegs
BUCKET_RADIUS   = 4   # size of the buckets
BUCKETS         = 24  # number of buckets
BUCKET_OFFS     = 30  # location of bucket distribution graph
COLLISION_DIST  = BALL_RADIUS + PEG_RADIUS # collision distance
COLLISION_DIST2 = COLLISION_DIST**2 # collision distance squared
PEG_SIZE        = 30  # distance between pegs
REBOUND         = 0.3 # rebound velocity factor     

now = lambda: int(round(time.time() * 1000))
logging.info(now())

# vector stuff
def dot(v1, v2): return v1[0]*v2[0] + v1[1]*v2[1]
def vlen(v): return math.sqrt(v[0]*v[0] + v[1]*v[1])

def reflect(d, n):
    '''
        Reflect vector d around vector n using: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
    '''
    # normalise n
    nlen = math.sqrt(n[0]*n[0] + n[1]*n[1])
    logging.debug('nlen {}'.format(nlen))
    n = [ n[0]/nlen, n[1]/nlen]
    
    # get the dot product d.n
    dotp = dot(d,n)
    
    # and reflect
    N = [ d[0] - 2*dotp*n[0], d[1] - 2*dotp*n[1] ]
    return N


class Ball:
    '''
        A ball
    '''    
    def __init__(self, uid, pos, vel):
        self.uid      = uid       # unique ball number
        self.pos      = pos       # location of the ball (x,y)
        self.vel      = vel       # velocity of the ball (vx, vy)
        self.prevpos  = pos
        self.enabled  = True

    def physics(self, delta):
        self.prevpos = self.pos.copy()
        self.pos[0] += self.vel[0] * delta
        self.pos[1] += self.vel[1] * delta
        self.vel[1] -= GFORCE * delta
                
    def drawball(self, ax):
#         self.image = plt.Circle(self.pos, BALL_RADIUS, color='b', fill=False)
        self.image = plt.Circle(self.pos, BALL_RADIUS, color='#555555', fill=True)
        ax.add_artist(self.image)
        return self.image
    
    def updateball(self):
#         self.image.center = self.pos
        return self.image
    
    def __str__(self):
        return "[ball #{} pos=({:.1f},{:.1f}) vel=({:.1f},{:.1f}) prev=({:.1f},{:.1f})]".format(
                  self.uid, self.pos[0], self.pos[1], self.vel[0], self.vel[1], self.prevpos[0], self.prevpos[1])

class Peg:
    '''
        A peg
    '''    
    def __init__(self, uid, pos):
        self.uid = uid      # unique peg number
        self.pos = pos       # location of the ball (x,y)
        
    def drawpeg(self, ax):
        self.image = plt.Circle(self.pos, PEG_RADIUS, color='g')
        ax.add_artist(self.image)
        return self.image
    
    def updatepeg(self):
        return self.image
    
    def __str__(self):
        return "[peg #{} pos=({:.1f},{:.1f})]".format(
                  self.uid, self.pos[0], self.pos[1])

class Bucket:
    '''
        A bucket
    '''    
    def __init__(self, uid, pos):
        self.uid   = uid      # unique number
        self.pos   = pos
        self.count = 0;       # number of hits
         
    def hit(self):
        self.count  += 1
        self.pos[1] -= 5

class Board:
    '''
        The simulator
    '''
    def __init__(self, rows, balls, ax):
        self.ax          = ax
        self.pegs        = []
        self.balls       = []
        self.buckets     = []
        self.bucketcount = 0
        
        # create pegs
        pos  = (0,0)
        n    = 0
        cols = 1
        DX   = self.hexw(PEG_SIZE)
        DY   = -self.hexv(PEG_SIZE)
        p    = ( pos[0], pos[1] )
        for i in range(0, rows):
            p = ( pos[0], pos[1] )
            for j in range(0, cols):
                peg = Peg(n, p)
                logging.info('add peg: {}'.format(peg))
                self.pegs.append(peg)
                p = ( p[0]+DX, p[1] )
                n += 1
            pos = ( pos[0] - DX/2, pos[1] + DY )
            cols += 1
        self.endpos = pos
        self.endlen = p[0]-self.endpos[0]
        logging.debug('end {} {}'.format(self.endpos, self.endlen))
        
        # create balls
        for i in range(0, balls):
            self.createball()
 
        # create buckets
        delta = self.endlen / BUCKETS
        logging.debug('buckets {} {} {}'.format(delta, self.endlen, self.endpos[0]))
        for i in range(0, BUCKETS):
            self.buckets.append(Bucket(len(self.buckets), [self.endpos[0] + i * delta, self.endpos[1] - BUCKET_OFFS]))
        
    
    def createball(self):
        ball = Ball(len(self.balls), [0,30], [random.uniform(-0.2,0.2),0])
        logging.info('add ball: {}'.format(ball))
        self.balls.append(ball)
        return ball
        
    def hexw(self, size): return math.sqrt(3) * size
    def hexh(self, size): return 2 * size
    def hexv(self, size): return 3.0/4.0*self.hexh(size)

    def refreshdistribution(self):
        x = np.asarray([ self.buckets[i].pos[0] for i in range(0, len(self.buckets)) ])
        y = np.asarray([ self.buckets[i].pos[1] for i in range(0, len(self.buckets)) ])
        
        # cubic fit
#         f = interp1d(x, y, kind='cubic')
#         fit = f(x)
        
        # polynomial fit
        z = np.polyfit(x, y, 4)
        f = np.poly1d(z)
        fit = f(x)

        # normal distr fit
#         mu, std = norm.fit(self.endpos[1]-y*1.0-BUCKET_OFFS)
#         print('m,s', mu, std)
#         fit = norm.pdf(x, mu, std*10)
#         fit = self.endpos[1]-fit*10000-BUCKET_OFFS
#         print('fit', fit)
        
        if hasattr(self, 'distrib'):
            self.distrib[0].set_data(x, y)
            self.distrib[1].set_data(x, fit)
        return x, y, x, fit
        
    def drawdistribution(self):
        x, y, z, f = self.refreshdistribution()
        self.distrib = plt.plot(x, y, 'b.', z, f, 'r-')
        return self.distrib
    
    def updatedistribution(self):
        return self.distrib
    
    def drawall(self):
        a = [ self.pegs[i].drawpeg(ax) for i in range(0,len(self.pegs)) ]
        a.extend([ self.balls[i].drawball(self.ax) for i in range(0,len(self.balls)) ])
        a.extend(self.drawdistribution())
        logging.debug('distrib: {}', str(a[len(a)-1]))
        return a # list of all pegs and balls
    
    def updateall(self, delta):
        # simulate physics
        for ball in self.balls:
            if not ball.enabled: continue
            ball.physics(delta)
#             logging.debug('physics {}'.format(ball))
            
            # if below the pegs we stick it in a bucket
            if ball.pos[1] < self.endpos[1]:
                # count the ball position
                idx = math.floor((ball.pos[0]-self.endpos[0]) / self.endlen * len(self.buckets) + 0.5)
                logging.debug('index: {} {} {} {}'.format(idx, self.endpos, self.endlen, len(self.buckets)))
                self.count(idx)
                
                # reset the ball (in-place for graphics)
                ball.pos[:] = [0,30]
                ball.vel[:] = [random.uniform(-0.2,0.2),0]

            self.collision(ball)
        
        # and redraw all balls
        a = [ self.pegs[i].updatepeg() for i in range(0,len(self.pegs)) ]
        a.extend([ self.balls[i].updateball() for i in range(0,len(self.balls)) ])
        a.extend(self.updatedistribution())
#         logging.debug('updateboard {}'.format(a))
        return a
    
    def count(self, idx):
        self.bucketcount += 1
        if idx >= 0 and idx < len(self.buckets):
            self.buckets[idx].hit()
            self.refreshdistribution()
        logging.debug('buckets: {} {}'.format(self.bucketcount, self.buckets))
        
    # determine the peg if any that a ball collides with
    def findcollision(self, ball):
        for peg in self.pegs:
            dx = peg.pos[0] - ball.pos[0]
            dy = peg.pos[1] - ball.pos[1]
            dist2 = dx*dx + dy*dy
            if dist2 < COLLISION_DIST2:
                return peg
        return None

    # calculate the new position of a ball colliding with a peg
    def calculate_rebound(self, peg, ball):
        logging.debug('rebound peg {}'.format(peg))
        L = [ ball.pos[0] - ball.prevpos[0], ball.pos[1] - ball.prevpos[1] ]
        logging.debug('L {}'.format(L))
        
        D = [ ball.prevpos[0] - peg[0], ball.prevpos[1] - peg[1] ]
        logging.debug('D {}'.format(D))
        
        # get a*t^2 + b*t + c
        a = L[0]*L[0]+L[1]*L[1]
        b = 2*(L[0]*D[0] + L[1]*D[1])
        c = D[0]*D[0] + D[1]*D[1] - COLLISION_DIST2
        logging.debug('abc {} {} {}'.format(a,b,c))
        
        # solve for t using quadratic equation
        t1 = (-b+math.sqrt(b*b-4*a*c)) / (2*a)
        t2 = (-b-math.sqrt(b*b-4*a*c)) / (2*a)
        t = t1 if t1 < t2 else t2
        logging.debug('t {} {} {}'.format(t1, t2, t))
        
        # get location of ball at time t which is the collision point
        C = [ t*L[0] + ball.prevpos[0], t*L[1] + ball.prevpos[1] ]
        logging.debug('C {}'.format(C))
#         cirle(C, BALL_RADIUS, 'g')
        
        # get the vector PC going from the peg to C that will be used to reflect around
        PC = [ C[0]-peg[0], C[1]-peg[1] ]
        logging.debug('PC {}'.format(PC))
#         vectrel(peg, PC, 'm')
        
        # get the vector CB which is the remainder of the vector prev-cur that we have to reflect
        PB = [ ball.pos[0] - C[0], ball.pos[1] - C[1] ]
        logging.debug('PB {}'.format(PB))
#         vectrel(peg, PB, 'm')
        
        # reflect PB around PC and invert it giving the final ball position
        # the formula is d=PB, n=PC: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
        N = reflect(PB, PC)
        logging.debug('N {}'.format(N))
#         vectrel(C, N, 'y')
        
        # calculate the new velocity with the direction of vector N and same velocity
        scale = vlen(ball.vel) / vlen(N) * REBOUND
        logging.debug('vel {}'.format(vlen(ball.vel)))
        logging.debug('scale {}'.format(scale))
        V = [ scale*N[0], scale*N[1] ]
        logging.debug('V {}'.format(V))
#         vectrel(C, V, 'black')
        
        # set the new position and velocity
        ball.pos[0] = C[0]+N[0] # in place for graphics
        ball.pos[1] = C[1]+N[1]
        ball.vel[0] = V[0]
        ball.vel[1] = V[1]
        
    # check for collision of a ball with any peg, and correct trajectory if necessary
    def collision(self, ball):
        # find a colliding peg
        peg = self.findcollision(ball)
        if not peg: return # no collision
        
        # calculate the corrected rebound position
        self.calculate_rebound(peg.pos, ball)
        logging.debug('collision {} {}'.format(ball, peg))

fig, ax = plt.subplots()
ax.figure.canvas.mpl_connect('key_press_event', lambda evt: sys.exit() if evt.key == 'escape' else None)
plt.title('GALTON')
ax.axis([-400, 400, -800, 100])
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

# create the board
board = Board(12, BALL_COUNT, ax)
prevtime = now()

# animate
def init():
    return board.drawall()

def animate(frame):
    global prevtime
    curtime = now()
    delta = (curtime - prevtime) / 1000.0
    prevtime = curtime
    a = board.updateall(delta)
    return a
 
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=10000, interval=50, blit=True)

# init()
logging.debug('show')
plt.show()
