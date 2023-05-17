import numpy as np

class ReachabilityMarginFunction3D:
    def __init__(self, ego_radius=0.15, obstacles=None, obstacle_radius=0.08):
        # Reachability margin function
        self.obstacle_center = obstacles 
        self.obstacle_radius = obstacle_radius
        self.ego_radius = ego_radius
    
    def eval(self, obs, action):
        obstacle_distances = np.linalg.norm(  obs[None, :, 0:2] - self.obstacle_center[:, None, :], axis=-1) - self.obstacle_radius - self.ego_radius
        obstacle_index = np.argmin(obstacle_distances, axis=0) 
        # Obstacle margins
        obstacle_margin = -1*obstacle_distances[obstacle_index]

        # Find maximum margin
        margin_index = 0
        margin_sub_index = obstacle_index
        return -1*obstacle_margin[0, 0], margin_index, margin_sub_index
    
    def dcdx(self, obs, action, margin_index, margin_subindex):
        dcdx_avoid = np.zeros(obs.shape)
        
        #First derivative computation
        if margin_index==0:
            dcdx_avoid_norm = np.linalg.norm( obs[:, 0:2] - self.obstacle_center[margin_subindex] )
            dcdx_avoid[:, 0:2] = -( obs[:, 0:2] - self.obstacle_center[margin_subindex] ) /dcdx_avoid_norm

        return -1*dcdx_avoid
    
    def dcdx2(self, obs, action, margin_index, margin_subindex):
        dcdx2s_avoid = np.zeros((obs.shape[1], obs.shape[1])) 

        # Second derivative computation
        vector_from_obstacle = obs[:, 0:2] - self.obstacle_center[margin_subindex]
        distance_from_obstacle = np.linalg.norm( vector_from_obstacle )

        # Second derivative of obstacle margin        
        dcdx2s_avoid[0:2, 0:2] = -1*np.eye(2)/distance_from_obstacle 
        dcdx2s_avoid[0:2, 0:2] += (1/(distance_from_obstacle)**3)*np.outer(vector_from_obstacle, vector_from_obstacle)

        return -1*np.array([dcdx2s_avoid]*obs.shape[0])