import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from copy import deepcopy

class SIM2D:
    def __init__(self) -> None:

        # initialize grid world
        self.grid_world = (12, 10)  # bounds of world
        self.fov = (4,3)    # 'robot' (x,y) field of view
        self.gaze_pos = None    # where the 'robot' is looking
        self._initialize_gaze()

        # action space
        # self.actions = ["up", "down", "right", "left", "no"]
        self.actions = [0, 1, 2, 3, 4]

        # initialize people locations and speakers
        self.speaker_fraction = 1/3
        self.num_people = 10
        self.num_speakers = None
        self.people_loc = None
        self.speakers_loc = None
        self._initialize_people()

    def _initialize_gaze(self):
        x = np.random.randint(0, self.grid_world[0])
        y = np.random.randint(0, self.grid_world[1])
        self.gaze_pos = (x,y)
    
    def _initialize_people(self):
        """Randomizes the number of people present and the number of speakers
            Creates those people as points on a gridworld"""
        # self.num_people = np.random.randint(5, 20)
        self.num_speakers = int(self.num_people*self.speaker_fraction)
        self._initialize_locations()
        self._initialize_speakers()

    def _initialize_locations(self):
        """Creates self.num_people number of random points within the total field of view"""
        x_locations = np.random.uniform(0, self.grid_world[0], size=(self.num_people, 1))
        y_locations = np.random.uniform(0, self.grid_world[1], size=(self.num_people, 1))
        is_speaker = np.zeros((self.num_people, 1))
        self.people_loc = np.hstack([x_locations, y_locations, is_speaker])
        
    def _initialize_speakers(self):
        """Grabs self.num_speakers number of random points from the people locations
            and assigns them as speakers"""
        # pick a self.num_speakers points from the people x-locations
        x_locs = np.random.choice(self.people_loc[:,0], size=(self.num_speakers,1), replace=False)
        # find the corresponding y-locations 
        speaker_indices = np.asarray([np.where(self.people_loc == x)[0][0] for x in x_locs])
        y_locs = self.people_loc[:,1][speaker_indices]
        y_locs = np.reshape(y_locs, x_locs.shape)
        self.speakers_loc = np.hstack([x_locs, y_locs])
        # for each speaker index, assign the relevant location in people_loc to a "1"
        self.people_loc[:,2][speaker_indices] = 1

    def sample_action_discrete(self):
        return np.random.choice(self.actions)
    
    def sample_action_continuous(self):
        """Currently (Δx, Δy) to move"""
        delx = np.random.uniform(0, 3)
        dely = np.random.uniform(0, 2)
        return delx, dely
    
    def return_state(self):
        state = np.ones((1, len(self.people_loc[0]), len(self.people_loc)))
        for i in range(len(self.people_loc[0])):
            state[0][i] = self.people_loc[:,i]

        return state

    def return_speakers(self):
        pass

    def reset(self):
        # self._initialize_people()
        self._initialize_gaze()
        return deepcopy(self.gaze_pos)

    def _is_valid(self, position):
        """Checks if the gaze position of the robot is within the grid world bounds"""
        x,y = position
        if x < 0 or x > self.grid_world[0]-1:
            return False
        if y < 0 or y > self.grid_world[1]-1:
            return False
        return True
    
    def _is_inside_fov(self, point, position):
        """Checks if one point is inside the robot's field of view"""
        x,y = point
        xmin = position[0] - self.fov[0]/2
        xmax = position[0] + self.fov[0]/2
        ymin = position[1] - self.fov[1]/2
        ymax = position[1] + self.fov[1]/2

        if x < xmin or x > xmax:
            return False
        if y < ymin or y > ymax:
            return False
        return True
          
    def _get_reward(self, position):
        reward = 0
        for point in self.people_loc:
            if self._is_inside_fov(point[0:2], position):
                # gets 3 points for each speaker in the field of view
                if point[2] == 1:
                    reward += 10
                # gets 1 point for each person in the field of view
                else:
                    reward += 2
            else:
                reward = -1
        return reward

    def _take_action(self, action, position):
        x,y = position
        if action == 0:
            y+=1
        elif action == 1:
            y-=1
        elif action == 2:
            x+=1
        elif action == 3:
            x-=1
        elif action == 4:
            pass
        if self._is_valid([x,y]):
            return [x,y]
        else:
            return position

    def step_discrete(self, action):
        self.gaze_pos = self._take_action(action, self.gaze_pos)
        reward = self._get_reward(self.gaze_pos)
        return deepcopy(self.gaze_pos), reward
    
    def move_continuous(self, position):
        self.gaze_pos = position
        reward = self._get_reward(self.gaze_pos)
        return deepcopy(self.gaze_pos), reward

    def make_plot(self, show=True):
        fig, ax = plt.subplots(1,1)
        # plot the people, speakers circled in red
        ax.scatter(self.people_loc[:,0], self.people_loc[:,1])
        ax.scatter(self.speakers_loc[:,0], self.speakers_loc[:,1], s=80, facecolors='none', edgecolors='r')

        # plot the 'robot' field of view centerpoint and square
        ax.plot(self.gaze_pos[0], self.gaze_pos[1], 'y+')
        rect_ll = (self.gaze_pos[0] - self.fov[0]/2, self.gaze_pos[1] - self.fov[1]/2)
        fov_rect = Rectangle(rect_ll, self.fov[0], self.fov[1], facecolor='none', edgecolor='y')
        ax.add_patch(fov_rect)

        ax.set_xlim(-1, self.grid_world[0])
        ax.set_ylim(-1, self.grid_world[1])
        
        if show:
            plt.show()

        return fig, ax

if __name__ == "__main__":
    mysim = SIM2D()

    # reward = mysim._get_reward(mysim.gaze_pos)
    # print(f"reward: {reward}")
    # mysim.make_plot()

    mysim.return_state()

    print(mysim.people_loc)

    
