from random import randint
from copy import deepcopy

class gridWorld:
    def __init__(self):
        self.grid_size = [5, 10]
        self.init_door=[9,1]
        self.actions = ["up","down","left","right"]
        self.agent=None
        self.door=None

    def reset(self):
        self.door=self.init_door
        position=[-1,-1]
        while not self.isValid(position):
            position[0]=randint(0,9)
            position[1]=randint(0,4)
        self.agent=position
        return deepcopy(self.agent)

    def take_action(self,action,position):
        x,y=position
        if action=="up":
            y+=1
        if action=="down":
            y-=1
        if action=="right":
            x+=1
        if action=="left":
            x-=1
        if self.isValid([x,y]):
            return [x,y]
        return position

    def step(self,action,rng_door=False):
        self.agent=self.take_action(action,self.agent)
        if rng_door:
            rng_action=self.actions[randint(0,3)]
            self.door=self.take_action(rng_action,self.door)
        if self.agent[0]==self.door[0] and self.agent[1]==self.door[1]:
            reward=20
        else:
            reward=-1
        return deepcopy(self.agent), reward

    def isValid(self,position):
        x,y=position
        if x<0 or x>9:      #out of x bounds
            return False
        if y<0 or y>4:      #out of y bounds
            return False
        if x==7 and y<3:    #if wall
            return False
        return True

if __name__=="__main__":
    #example usage for a gym-like environment 
    #state: [x,y] coordinate of the agent
    #actions: ["up","down","left","right"] directions the agent can move
    env=gridWorld()
    for learning_epoch in range(100):
        state=env.reset()                        #every episode, reset the environment to the original configuration
        for time_step in range(20):
            action=env.actions[randint(0,3)] #learner chooses one of these actions
            next_state,reward=env.step(action)  #the action is taken, a reward and new state is returned
            #note: use env.step(action,rng_door=True) for part 2

            