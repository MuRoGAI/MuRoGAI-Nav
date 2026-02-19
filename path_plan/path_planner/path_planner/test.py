import numpy as np


grid = np.load('restaurant.npy')
res = 0.1

height, width = grid.shape
height, width = height*res, width*res

# print("-"*60)
# print("|\n"*10)
print(f"Grid width {width} ")
print(f"Grid height {height} ")
# print("|\n"*10)
# print("-"*60)

bounds = (np.array([0.0, 0.0]), np.array([height, width]))
W = int(width / res)
H = int(height / res)


def find_Pstar(positions, yaw):
    p = []
    centroid = np.mean(positions, axis=0)
    Rinv = np.array([[np.cos(yaw), np.sin(yaw)],[-np.sin(yaw), np.cos(yaw)]])
    print(centroid)
    h,_ = np.shape(positions)
    print(Rinv)
    for i in range(h):
        
        delta = positions[i] - centroid
        print(delta)
        pistar = Rinv@np.transpose(delta)
        print(pistar)
        p.append(pistar.tolist())
    return p
        
positions = np.array([[4,6],[4,4],[5,5]])
p = find_Pstar(positions,yaw=0)
print(p)



def find_sx_sy(P_star,radius):
    D = []
    p = 10000
    I = 10000
    k = 0
    for i in P_star:
        d = np.sqrt(i[0]**2+i[1]**2)
        if d<p:
           p = d
           I = k
        
        k = k+1
           
    X,Y = P_star[int(I)]

    sy = radius/(np.sqrt(X**2+Y**2)) ### write in paper
    sx = radius/(np.sqrt(X**2+Y**2))

    
    return sx,sy

     
sx,sy = find_sx_sy(p,1)
print(sx,sy)
# def disc_collides(x: float, y: float, radius: float) -> bool:
#     """Check if disc overlaps any obstacle. Uses Numba if available."""
#     if NUMBA_AVAILABLE:
#         return _nb_disc_collides(
#             x, y, radius, grid, seresolution,
#             origin[0], origin[1], self.H, self.W
#         )
#     # Fallback to pure Python
#     r = float(radius)
#     i0, j0 = world_to_cell(x - r, y - r)
#     i1, j1 = world_to_cell(x + r, y + r)
#     i0 -= 1; j0 -= 1
#     i1 += 1; j1 += 1
#     rr = r * r
#     for i in range(i0, i1 + 1):
#         for j in range(j0, j1 + 1):
#             if not self.in_bounds(i, j):
#                 return True
#             if self.grid[i, j] == 0:
#                 continue
#             cx = self.origin[0] + (j + 0.5) * self.resolution
#             cy = self.origin[1] + (i + 0.5) * self.resolution
#             if (cx - x) ** 2 + (cy - y) ** 2 <= rr:
#                 return True
#     return False

from si_rrt_enhanced_individual_kinodynamic import OccupancyGrid

def find_scaling(P_star,grid,centroid,radius):
    
    OccupancyGrid.disc_collides(centroid[0],centroid[1])