from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
##########################################################
##           GLOGAL          PARAMETRS 
##########################################################
STEP =0.08
W = 500
H = 500
light_point = np.array([-1,1,0])
##########################################################

##########################################################
##           SCREEN SETTINGS 
##########################################################
SCREEN = np.zeros((H,W,3))
W_space = 1.5
H_space = float(H/W)*W_space
arr_w = np.linspace(-W/2,W/2,W)
arr_h = np.linspace(-H/2,H/2,H)
##########################################################


def Get_coord_space(i,j):
	Z = 0.75
	return np.array([i*W_space/W,j*H_space/H,Z])

def normalize(x):
	x /= np.linalg.norm(x)
	return x


class Plane( object ):
	def __init__(self, point, normal, color,id_):
		self.n = normal/np.linalg.norm(normal)
		self.p = point
		self.color = color
		self.type = "Plane"
		self.id = id_

	def intersection(self, point_cur,point_next):
		A = (point_cur-self.p).dot(self.n)
		B = (point_next-self.p).dot(self.n) 
		if A*B>0:
			return False
		else:
			return True
	def Normal(self,point):
		return self.n

class Sphere(object):
	def __init__(self, center,R,color,id_):
		self.center = center
		self.R = R
		self.color = color
		self.type = "Sphere"
		self.id = id_

	def intersection(self, point_cur,point_next):
		A = (point_cur-self.center).dot(point_cur-self.center) - self.R**2
		B = (point_next-self.center).dot(point_next-self.center)- self.R**2
		if A*B>0:
			return False
		else:
			return True
	def Normal(self,point):
		return normalize(point - self.center)

class Ray( object):
	def __init__(self, start_point, direct_vector):
		self.start = start_point
		self.direct_vector = direct_vector
		self.t = 0
		self.curr_point = start_point
	def Next(self):
		self.t+=STEP
		self.curr_point = self.start+self.direct_vector*self.t
		return self
	def __str__(self):
		return self.curr_point.__str__()


def INTERSECTION(ray,SCENE,depth):
	for q in range(depth):
		ray_curr = ray.curr_point 
		ray_next = ray.Next().curr_point
		for obj in SCENE:
			if obj.intersection(ray_curr,ray_next):
				return [True,ray_curr,obj.id]
	return [False,False,False]

def Get_intens(x):
	if x >=0:
		return x
	else:
		return 0

sph_1 = Sphere(np.array([0,0,2]),0.5,np.array([0,0,1]),0)
pln_1 = Plane(np.array([0,0,3]),np.array([0,0,-1]),np.array([1,1,1]),1)
sph_2 = Sphere(np.array([-1,1,2]),0.5,np.array([1,0,0]),2)
SCENE = [sph_1,pln_1,sph_2]


##########################################################
## MAIN ALGHORITHM
########################################################## 

# loop for all pixels
for i, y in enumerate(arr_h):
	for j, x in enumerate(arr_w):
		direct =normalize(Get_coord_space(x,y))
		ray = Ray(direct,direct)
		flag,collision_point, index = INTERSECTION(ray,SCENE,200)
		if flag:  # first collision
			ray_2 = Ray(collision_point,normalize(light_point-collision_point))
			if INTERSECTION(ray_2,SCENE,200)[0] == False:
				intensivity = Get_intens(normalize(light_point-collision_point).dot(SCENE[index].Normal(collision_point)))
			else:
				intensivity =0.01
			SCREEN[H-i-1][j][0] = SCENE[index].color[0]*intensivity 
			SCREEN[H-i-1][j][1]	= SCENE[index].color[1]*intensivity
			SCREEN[H-i-1][j][2] = SCENE[index].color[2]*intensivity

	# print progress
	print(float(i*W+j)/(W*H)*100,"%")					

plt.imshow(SCREEN)
plt.show()



