'''
Galton board simulator.
Support math.
    
Kenneth Larsen
kenbolarsen@gmail.com

'''

import sys, math, random, time
import matplotlib.pyplot as plt

GFORCE          = 9.8/10 # gravity
BALL_RADIUS     = 50   # size of the balls
PEG_RADIUS      = 50   # size of the pegs
COLLISION_DIST2 = (BALL_RADIUS + PEG_RADIUS)**2 # collision distance squared
PEG_SIZE        = 30     

now = lambda: int(round(time.time() * 1000))
print(now())

fig, ax = plt.subplots()
ax.axis([-250, 250, -250, 250])
ax.set_aspect('equal')

def cirle(pos, r, c):
    image = plt.Circle(pos, r, color=c, fill=False)
    ax.add_artist(image)

def vectrel(start, size, col):
    return ax.quiver(start[0], start[1], size[0], size[1], units='xy', scale=1, color=col)

def vect(start, end, col):
    vectrel(start, [ end[0]-start[0], end[1]-start[1] ], col)

class Ball:
    '''
     A ball
    '''    
    def __init__(self, uid, pos, vel):
        self.uid      = uid       # unique ball number
        self.pos      = pos       # location of the ball (x,y)
        self.vel      = vel       # velocity of the ball (vx, vy)
        self.prevpos  = pos
    
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
    
    def __str__(self):
        return "[peg #{} pos=({:.1f},{:.1f})]".format(
                  self.uid, self.pos[0], self.pos[1])

# vector stuff
def dot(v1, v2): return v1[0]*v2[0] + v1[1]*v2[1]
def vlen(v): return math.sqrt(v[0]*v[0] + v[1]*v[1])
def vlen2(b,e): return vlen([ e[0]-b[0], e[1]-b[1] ])

def reflect(d, n):
    # normalise n
    nlen = math.sqrt(n[0]*n[0] + n[1]*n[1])
    print('nlen', nlen)
    n = [ n[0]/nlen, n[1]/nlen]
    
    # get the dot product d.n
    dotp = dot(d,n)
    
    # and reflect
    N = [ d[0] - 2*dotp*n[0], d[1] - 2*dotp*n[1] ]
    return N

# calculate the new position of a ball colliding with a peg
def calculate_rebound(peg, ball):
    print('rebound peg', peg)
    L = [ ball.pos[0] - ball.prevpos[0], ball.pos[1] - ball.prevpos[1] ]
    print('L', L)
#         if L[0] < 0.001 and L[1] < 0.001: return sys.exit(1)
    
    D = [ ball.prevpos[0] - peg[0], ball.prevpos[1] - peg[1] ]
    print('D', D)
    
    # get a*t^2 + b*t + c
    a = L[0]*L[0]+L[1]*L[1]
    b = 2*(L[0]*D[0] + L[1]*D[1])
    c = D[0]*D[0] + D[1]*D[1] - COLLISION_DIST2
    print('abc', a,b,c)
    
    # solve for t using quadratic equation
    t1 = (-b+math.sqrt(b*b-4*a*c)) / (2*a)
    t2 = (-b-math.sqrt(b*b-4*a*c)) / (2*a)
    t = t1 if t1 < t2 else t2
    print('t', t1, t2, t)
    
    # get location of ball at time t which is the collision point
    C = [ t*L[0] + ball.prevpos[0], t*L[1] + ball.prevpos[1] ]
    print('C', C)
    cirle(C, BALL_RADIUS, 'g')
    
    # get the vector PC going from the peg to C that will be used to reflect around
    PC = [ C[0]-peg[0], C[1]-peg[1] ]
    print('PC', PC)
    vectrel(peg, PC, 'm')
    
    # get the vector CB which is the remainder of the vector prev-cur that we have to reflect
    PB = [ ball.pos[0] - C[0], ball.pos[1] - C[1] ]
    print('PB', PB)
    vectrel(peg, PB, 'm')
    
    # reflect PB around PC and invert it giving the final ball position
    # the formula is d=PB, n=PC: 𝑟=𝑑−2(𝑑⋅𝑛)𝑛
    N = reflect(PB, PC)
    print('N', N)
    vectrel(C, N, 'y')
    
    # calculate the new velocity with the direction of vector N and same velocity
    scale = vlen(ball.vel) / vlen(N)
    print('vel', vlen(ball.vel))
    print('scale', scale)
    V = [ scale*N[0], scale*N[1] ]
    print('V', V)
    vectrel(C, V, 'black')

# test down on the right   
B = Ball(1, [60,10], [10,10])
B.prevpos = [30, 150]
P = Peg(1, [0,0])
# test down on the left   
B = Ball(1, [-60,10], [10,10])
B.prevpos = [30, 150]
P = Peg(1, [0,0])
# test up on the left   
B = Ball(1, [-60,10], [10,10])
B.prevpos = [30, -150]
P = Peg(1, [0,0])
# test up on the right   
B = Ball(1, [60,10], [10,10])
B.prevpos = [30, -150]
P = Peg(1, [0,0])
# test in from the upper right   
B = Ball(1, [60,30], [10,10])
B.prevpos = [150, 30]
P = Peg(1, [0,0])
# test in from the upper left   
B = Ball(1, [60,30], [10,10])
B.prevpos = [-150, 30]
P = Peg(1, [0,0])
# test in from the lower left   
# B = Ball(1, [60,-30], [10,10])
# B.prevpos = [-150, -30]
# P = Peg(1, [0,0])
# test in from the lower right   
# B = Ball(1, [60,-30], [10,10])
# B.prevpos = [150, -30]
# P = Peg(1, [0,0])

cirle(B.pos, BALL_RADIUS, 'b')
cirle(B.prevpos, BALL_RADIUS, 'r')
cirle(P.pos, PEG_RADIUS, 'y')
vect(B.prevpos, B.pos, 'c')

calculate_rebound(P.pos, B)

plt.show()
