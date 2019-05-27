from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


##########################################################
##           GLOGAL          PARAMETRS 
##########################################################
STEP =0.1
W = 100
H = 100
light_point = np.array([-0.5,-0.5,0])
##########################################################

##########################################################
##           SCREEN SETTINGS 
##########################################################
SCREEN = np.zeros((H,W,3))
W_space = 1.3
H_space = float(H/W)*W_space
arr_w = np.linspace(-W/2,W/2,W)
arr_h = np.linspace(-H/2,H/2,H)
##########################################################


def Get_coord_space(i,j):
	Z = 1
	return np.array([j*H_space*2/H,i*W_space*2/W,Z])

def normalize(x):
	x /= np.linalg.norm(x)
	return x


class Plane( object ):
	def __init__(self, point, normal, color,id_):
		if abs(normal[0]**2+normal[1]**2+normal[2]**2-1)>=0.0001:
			self.n = normalize(normal)
		else:
			self.n = normal
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



sph_1 = Sphere(np.array([0,0,2]),1,np.array([1,0,0]),0)
pln_1 = Plane(np.array([0,0,5]),np.array([0,0,1]),np.array([1,1,1]),1)

SCENE = [sph_1,pln_1]

##########################################################
## MAIN ALGHORITHM
########################################################## 


# loop for all pixels
for i, x in enumerate(arr_h):
	for j, y in enumerate(arr_w):
		direct =normalize(Get_coord_space(x,y))
		ray = Ray(np.array([0,0,0]),direct)
		flag,collision_point, index = INTERSECTION(ray,SCENE,120)
		if flag:  # first collision
			ray_2 = Ray(collision_point,normalize(light_point-collision_point))
			if INTERSECTION(ray_2,SCENE,120)[0] == False:
				intensivity = abs((normalize(light_point-collision_point)).dot(SCENE[index].Normal(collision_point)))
			else:
				intensivity =0.1
			SCREEN[i][j][0] =  SCENE[index].color[0]*intensivity 
			SCREEN[i][j][1]	= SCENE[index].color[1]*intensivity
			SCREEN[i][j][2] = SCENE[index].color[2]*intensivity

	# print progress
	print(float(i*W+j)/(W*H)*100,"%")					


plt.imshow(SCREEN)
plt.show()












		# cross = False
		# for q in range(102):
		# 	if(cross):
		# 		break
		# 	ray_curr = ray.curr_point 
		# 	ray_next = ray.Next().curr_point
		# 	for obj in SCENE:
		# 		if obj.intersection(ray_curr,ray_next):
		# 			SCREEN[i][j][0] = obj.color[0] 
		# 			SCREEN[i][j][1]	= obj.color[1]
		# 			SCREEN[i][j][2] = obj.color[2]
		# 			cross = True
		# 			break