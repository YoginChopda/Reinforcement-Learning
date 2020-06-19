# qlearningGhostAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningGhostAgents import ReinforcementGhostAgent
from ghostfeatureExtractors import *
import sys
import random,util,math
import pickle
from util import manhattanDistance
from game import Actions
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras.models import load_model

class QLearningGhostAgent(ReinforcementGhostAgent):
    """
      Q-Learning Ghost Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self,epsilon=1,gamma=0.95,alpha=0.001, numTraining=0,agentIndex=1, extractor='GhostIdentityExtractor', **args):
        "You can initialize Q-values here..."
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        args['agentIndex'] = agentIndex
        self.index = agentIndex
        self.memory = deque(maxlen=2000)
        self.q_values = util.Counter()
        self.act_dict={0:'North',
                       1:'South',
                       2:'West',
                       3:'East'
                       }
        self.action_size=4 #used for output layer in dqn
        self.state_size=12 #used for input layer in dqn
        self.batch_size=32 #used for model training
        self.featExtractor = util.lookup(extractor, globals())()
        self.weights = util.Counter()
        self.learning_rate = alpha
        self.gamma = gamma                # discount rate
        self.epsilon = epsilon               # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
#        self.path='D:/SMU/Courses/Introduction to AI/MiniProject_v2/MiniProject/Pacman_MiniProject'
        self.model = load_model('agent1_dqn.h5')
#        self.model=self._build_model()
        self.safety_distance = 12                                      # safety distance (capsule to pacman + ghost to pacman)
        self.last_node_pacman_was_heading_towards = None
        self.last_node_pacman_was_moving_away_from = None
        self.last_action = None                                        # store action for Qvalues since action is defined differently
        #self.q_values = util.Counter()
#        self.q_values = self.load_obj("agent"+str(agentIndex)+"Q")    #load previous q values
       
        self.segment_dic = {
            1:[(1,6),(1,7),(1,8)],
            2:[(1,4),(1,3),(1,2)],
            3:[(2,9),(3,9),(4,9),(4,8)],
            4:[(2,5)],
            5:[(2,1),(3,1),(4,1),(4,2)],
            6:[(3,6),(3,7)],
            7:[(3,3),(3,4)],
            8:[(4,5),(5,5)],
            9:[(5,7)],
            10:[(5,3)],
            11:[(6,8),(6,9),(7,9),(8,9),(9,9),(10,9),(11,9),(12,9),(13,9),(13,8)],
            12:[(6,6)],
            13:[(6,4)],
            14:[(6,2),(6,1),(7,1),(8,1),(9,1),(10,1),(11,1),(12,1),(13,1),(13,2)],
            15:[(7,7),(8,7)],
            16:[(7,3),(8,3),(9,3),(10,3),(11,3),(12,3)],
            17:[(8,5),(9,5),(10,5),(11,5),(9,6),(10,6)],
            18:[(11,7),(12,7)],
            19:[(13,6)],
            20:[(13,4)],
            21:[(14,7)],
            22:[(14,5),(15,5)],
            23:[(14,3)],
            24:[(15,8),(15,9),(16,9),(17,9)],
            25:[(15,2),(15,1),(16,1),(17,1)],
            26:[(16,6),(16,7)],
            27:[(16,3),(16,4)],
            28:[(17,5)],
            29:[(18,8),(18,7),(18,6)],
            30:[(18,2),(18,3),(18,4)],
            #node
            31:[(1,9)],
            32:[(1,5)],
            33:[(1,1)],
            34:[(3,5)],
            35:[(4,7)],
            36:[(4,3)],
            37:[(6,7)],
            38:[(6,5)],
            39:[(6,3)],
            40:[(9,7)],
            41:[(10,7)],
            42:[(13,7)],
            43:[(13,5)],
            44:[(13,3)],
            45:[(15,7)],
            46:[(15,3)],
            47:[(16,5)],
            48:[(18,9)],
            49:[(18,5)],
            50:[(18,1)]
        }

        self.choice_dic = {
            1:(31,32),
            2:(32,33),
            3:(31,35),
            4:(32,34),
            5:(33,36),
            6:(34,35),
            7:(34,36),
            8:(34,38),
            9:(35,37),
            10:(36,39),
            11:(37,42),
            12:(37,38),
            13:(38,39),
            14:(39,44),
            15:(37,40),
            16:(39,44),
            17:(40,41),
            18:(41,42),
            19:(42,43),
            20:(43,44),
            21:(42,45),
            22:(43,47),
            23:(44,46),
            24:(45,48),
            25:(46,50),
            26:(45,47),
            27:(46,47),
            28:(47,49),
            29:(48,49),
            30:(49,50),
            31:(32,35),
            32:(31,33,34),
            33:(32,36),
            34:(32,35,36,38),
            35:(31,34,37),
            36:(33,34,39),
            37:(35,38,40,42),
            38:(34,37,39),
            39:(36,38,44),
            40:(37,41),
            41:(40,42),
            42:(37,41,43,45),
            43:(42,44,47),
            44:(39,43,46),
            45:(42,47,48),
            46:(44,47,50),
            47:(43,45,46,49),
            48:(45,49),
            49:(47,48,50),
            50:(46,49)
        }
        
        
        ReinforcementGhostAgent.__init__(self, **args)
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        pass
        
    def computeActionFromQValues(self, state):
        pass
        
    def getWeights(self):
        return self.weights

#    def getQValue(self, state, action):
#        pass

        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        
        #~~~~~~~~
        #<Todo> Write your code here
        #~~~~~~~~
        model.add(Dense(40,activation='relu'))
        model.add(Dense(self.action_size))
        
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def get_value(self,action):
        if action=='North':
            return 0
        elif action=='South':
            return 1
        elif action=='West':
            return 2
        elif action=='East':
            return 3
        
    def update(self, state, action, nextState, reward):
#        print('Entered update stage')
#        print('action in update stage:',action)
        if self.isInTraining():
#            print(self.episodesSoFar)
            self.memory.append((self.getmystate(state), self.get_value(action), reward, self.getmystate(nextState)))
    #        print('nextstate:',self.getmystate(nextState))
            if len(self.memory)>self.batch_size:
    #            print('Training neural net')
                minibatch = random.sample(self.memory, self.batch_size)
                
                for st, act, rew, next_st in minibatch:
#                    print(st,act,rew,next_st)
#                    print('episode has not ended ..')
                    target = (rew + self.gamma *
                              np.amax(self.model.predict(np.reshape(next_st,[1,self.state_size]))[0]))
                    target_f = self.model.predict(np.reshape(st,[1,self.state_size]))
                    target_f[0][self.get_value(action)] = target
    #                print('model fit..')
                    self.model.fit(np.reshape(st,[1,self.state_size]), target_f, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
        else:
            print('Testing starts and no update..')
            
            
#            
#    def load(self, name):
#        self.load_model(name)

    def save(self, name):
        self.model.save(name)
        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        ReinforcementGhostAgent.final(self, state) 
        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            
            # you might want to print your weights here for debugging
            if self.agentIndex == 1:
                self.save("agent1_dqn.h5")
            
            
    def getAction(self, state):
        #Uncomment the following if you want one of your agent to be a random agent.
        if self.agentIndex == 1:
            if np.random.rand() <= self.epsilon:   # epsilon greedy action
                action= random.choice(self.getLegalActions(state))
            
            elif self.check_in_centre_left1(state):
                        action='East'
            elif self.check_in_centre_right1(state):
                action= 'West'
            elif self.check_in_centre_centre1(state):
                action= 'North'
            
            else:    
                act_values = self.model.predict(np.reshape(self.getmystate(state),[1,self.state_size]))
#                print('model_pred:',act_values)
                best_act_index=np.argmax(act_values[0])
                action=self.act_dict[best_act_index]
#                print('action from neural net:',action)
#                print('legal_actions:',self.getLegalActions(state))
                
                if action not in self.getLegalActions(state):
                    
                    if self.check_in_centre_left1(state):
                        action='East'
                    elif self.check_in_centre_right1(state):
                        action= 'West'
                    elif self.check_in_centre_centre1(state):
                        action= 'North'
                    else:
                        
    #                    print('taking random action')
    #                    action= random.choice(self.getLegalActions(state))
                        ghostState = state.getGhostState(1)
            #            print('ghostState:',ghostState)
                        legalActions = state.getLegalActions(1)
    #                    print('legalActions:',legalActions)
                        
                        pos = state.getGhostPosition(1)
                        isScared = ghostState.scaredTimer > 0
                
                        speed = 1
                        if isScared: speed = 0.5
                
                        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
    #                    print('actionvectors:',actionVectors)
                        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
                        pacmanPosition = state.getPacmanPosition()
                
                        # Select best actions given the state
              
                        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
                        if isScared:
                            bestScore = max( distancesToPacman )
            #                bestProb = self.prob_scaredFlee
                        else:
                            bestScore = min( distancesToPacman )
            #                bestProb = self.prob_attack
                        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
    #                    print('bestact:',bestActions)
                        
                        action=random.choice(bestActions)
    #                    print('random action:',action)
                else:
                    print('best action from neural net')
            self.doAction(state,action)
#            print('mystate:',self.getmystate(state))
        else:
#            if self.check_in_centre2(state):
#                action=random.choice(self.getLegalActions(state))
#            
#            else:
#                ghostState = state.getGhostState(2)
#    #            print('ghostState:',ghostState)
#                legalActions = state.getLegalActions(2)
#                print('legalActions:',legalActions)
#                
#                pos = state.getGhostPosition(2)
#                isScared = ghostState.scaredTimer > 0
#        
#                speed = 1
#                if isScared: speed = 0.5
#        
#                actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
#                print('actionvectors:',actionVectors)
#                newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
#                pacmanPosition = state.getPacmanPosition()
#        
#                # Select best actions given the state
#                distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
#                if isScared:
#                    bestScore = max( distancesToPacman )
#    #                bestProb = self.prob_scaredFlee
#                else:
#                    bestScore = min( distancesToPacman )
#    #                bestProb = self.prob_attack
#                bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]
#                print('bestact:',bestActions)
#                
#                action=random.choice(bestActions)
#                    
#            print('action for agent2:',action)
        #smart ghost
            node = self.chose_node_to_move_to(state)
            action = self.move_to_node(state,node)
        
            if self.is_same_segment_as_pacman(state) and not self.isScared(state):
                action = self.catch_pacman(state)
        
        return action

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

###############################################################################################
##Get features    
###############################################################################################
    def isScared1(self, state):    
        ghostState = state.getGhostState(1)
        if ghostState.scaredTimer > 0:
            return 1
        else:
            return 0

    
    def pacman_to_capsule_distance1(self, state):
        
        capsule_pos = state.getCapsules()
        pacman_pos = state.getPacmanPosition()
        
        dist_list = []
        for pos in capsule_pos:
            dist_list.append(manhattanDistance(pos, pacman_pos))

        if dist_list:
            distance = min(dist_list)
        else:
            distance = 99                                      #if dist_list is empty, both capsules have been eaten so set distance at arbitrary value

        return distance/100


    def dist_to_capsule1(self,state):
        capsule_pos = state.getCapsules()
        my_pos = state.getGhostPosition(1)
        
        dist_list = []
        for pos in capsule_pos:
            dist_list.append(manhattanDistance(pos, my_pos))

        if dist_list:
            distance = min(dist_list)
        else:
            distance=99
        return distance/100


    def dist_to_pacman1(self, state):
        pacman_pos = state.getPacmanPosition()
        my_pos = state.getGhostPosition(1)
        return (manhattanDistance(pacman_pos, my_pos))/100
     
    def same_lane1(self, state):
        pacman = state.getPacmanState()
        if self.get_segment(pacman.getPosition()) == self.get_segment(state.getGhostPosition(1)):
            return 1
        else:
            return 0
    
    def get_relative_x1(self, state):
        gy2, gx2 = state.getGhostPosition(2)
        gy, gx = state.getGhostPosition(1)
        
        return (gx-gx2)/100
        
    def get_relative_y1(self, state):
        gy2, gx2 = state.getGhostPosition(2)
        gy, gx = state.getGhostPosition(1)
        
        return (gy-gy2)/100

    def check_in_centre_left1(self, state):
        if state.getGhostPosition(1) in [(8,5)]:
            return 1
        else:
            return 0


    def check_in_centre_right1(self, state):
        if state.getGhostPosition(1) in [(11,5)]:
            return 1
        else:
            return 0
        

    def check_in_centre_centre1(self, state):
        if state.getGhostPosition(1) in [(9,5),(10,5),(9,6),(10,6)]:
            return 1
        else:
            return 0        
        
    def get_relative_x_pacman1(self, state):
        pacman = state.getPacmanState()
        py, px = pacman.getPosition()
        gy, gx = state.getGhostPosition(1)
        
        return (px-gx)/100
    
    def get_relative_y_pacman1(self, state):
        pacman = state.getPacmanState()
        py, px = pacman.getPosition()
        gy, gx = state.getGhostPosition(1)
        return (py-gy)/100
        
    def getmystate(self,state):
        mystate = (
            self.get_relative_x1(state),
            self.get_relative_y1(state),
            self.get_relative_x_pacman1(state),
            self.get_relative_y_pacman1(state),
            self.check_in_centre_left1(state),
            self.check_in_centre_right1(state),
            self.check_in_centre_centre1(state),
            self.same_lane1(state),
            self.isScared1(state),
            self.pacman_to_capsule_distance1(state),
            self.dist_to_capsule1(state),
            self.dist_to_pacman1(state)
            )
       
        return mystate

    def check_in_centre2(self, state):
            if state.getGhostPosition(2) in [(8,5),(9,5),(10,5),(11,5),(9,6),(10,6),(9,7),(10,7)]:
                return 1
            else:
                return 0    




#
#
##-----------------SMART GHOST CODE (NON-QLEARNING)---------------------------------------------------------------
# 
    
        
    
    
    
    def isScared(self, state):    
        ghostState = state.getGhostState(2)
        if ghostState.scaredTimer > 0:
            return True
        else:
            return False

    
    def pacman_to_capsule_distance(self, state):
        
        capsule_pos = state.getCapsules()
        pacman_pos = state.getPacmanPosition()
        
        dist_list = []
        for pos in capsule_pos:
            dist_list.append(manhattanDistance(pos, pacman_pos))

        if dist_list:
            distance = min(dist_list)
        else:
            distance = 999                                       #if dist_list is empty, both capsules have been eaten so set distance at arbitrary value

        return distance


    def dist_to_capsule(self,state):
        capsule_pos = state.getCapsules()
        my_pos = state.getGhostPosition(self.index)
        
        dist_list = []
        for pos in capsule_pos:
            dist_list.append(manhattanDistance(pos, my_pos))

        if dist_list:
            distance = min(dist_list)

        return distance


    def dist_to_pacman(self, state):
        pacman_pos = state.getPacmanPosition()
        my_pos = state.getGhostPosition(self.index)
        return manhattanDistance(pacman_pos, my_pos)
    

    def get_segment(self,pos):
        x, y = pos
        for segment in range(1,len(self.segment_dic)+1):
            if (math.ceil(x),math.ceil(y)) in self.segment_dic[segment]:
                return segment


    def get_my_nodes(self,state):
        my_segment = self.get_segment(state.getGhostPosition(2)) 
        my_nodes = self.choice_dic[my_segment]
        return my_nodes


    def get_pacman_nodes(self,state):
        pacman_segment = self.get_segment(state.getPacmanPosition())    
        pacman_nodes = self.choice_dic[pacman_segment]                  
        if len(pacman_nodes) > 2:                                         
            pacman_nodes = [pacman_segment, pacman_segment]
        return pacman_nodes


    def chose_node_to_move_to(self,state):

        pacman_node1, pacman_node2 = self.get_pacman_nodes(state)   
        pacman_node1_pos = self.segment_dic[pacman_node1][0]          
        pacman_node2_pos = self.segment_dic[pacman_node2][0]
 
        #FOR RUNNING 2 SMART GHOSTS

        '''   
        x1, y1 = state.getGhostPosition(1)
        x2, y2 = state.getGhostPosition(2)

        ghost1_pos = (math.ceil(x1), math.ceil(y1))
        ghost2_pos = (math.ceil(x2), math.ceil(y2))

        if manhattanDistance(ghost1_pos, pacman_node1_pos) + manhattanDistance(ghost2_pos, pacman_node2_pos) < manhattanDistance(ghost2_pos, pacman_node1_pos) + manhattanDistance(ghost1_pos, pacman_node2_pos):
            ghost1_pacman_node_pos = pacman_node1_pos
            ghost2_pacman_node_pos = pacman_node2_pos
        else:
            ghost1_pacman_node_pos = pacman_node2_pos
            ghost2_pacman_node_pos = pacman_node1_pos
        
        if self.agentIndex == 1:
            my_target_node_pos = ghost1_pacman_node_pos
        elif self.agentIndex == 2:
            my_target_node_pos = ghost2_pacman_node_pos

        '''
        #########################

        #FOR RUNNING 1 SMART GHOSTS (target closest node)

        my_pos = state.getGhostPosition(2)
        if manhattanDistance(my_pos, pacman_node1_pos) < manhattanDistance(my_pos, pacman_node2_pos):
            my_target_node_pos = pacman_node1_pos
        else:
            my_target_node_pos = pacman_node2_pos      

        #########################

        my_nodes = self.get_my_nodes(state)  

        node_list = []

        for node in my_nodes:
            node_pos = self.segment_dic[node][0]
            distance_from_node_to_target_node = manhattanDistance(node_pos, my_target_node_pos)
            node_list.append([distance_from_node_to_target_node, node])

        ghostState = state.getGhostState(2)

        if self.isScared(state) and self.dist_to_pacman(state)/ghostState.scaredTimer <= 3:   
            if state.getGhostPosition(2)[0] < state.getPacmanPosition()[0]:
                return 35
            else:
                return 46
        
        elif self.pacman_to_capsule_distance(state) + self.dist_to_pacman(state) < self.safety_distance and self.dist_to_capsule(state) > self.pacman_to_capsule_distance(state):

            return max(node_list)[1]    #if might get eaten, choose furthest node to move to

        else:        
            return min(node_list)[1]    #if won't get eaten, choose nearest node to move to



    def move_to_node(self,state,node):
        
        ghostState = state.getGhostState(2)
        legalActions = state.getLegalActions(2)
        pos = state.getGhostPosition(2)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        nodePosition = self.segment_dic[node][0]

        # Select best actions given the state
        distancesToNode = [manhattanDistance( pos, nodePosition ) for pos in newPositions]
        bestScore = min( distancesToNode )
        bestActions = [action for action, distance in zip( legalActions, distancesToNode ) if distance == bestScore]
        action = random.choice(bestActions)

        return action


    def is_same_segment_as_pacman(self,state):
        
        pacman_segment_list = list(self.get_pacman_nodes(state))
        pacman_segment_list.append(self.get_segment(state.getPacmanPosition()))
        my_segment = self.get_segment(state.getGhostPosition(2))

        if my_segment in pacman_segment_list:
            return True
        else:
            return False
   
    
    def catch_pacman(self,state):
    
        legalActions = state.getLegalActions(2)
        pos = state.getGhostPosition(2)

        speed = 1

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        bestScore = min( distancesToPacman )
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman) if distance == bestScore]
        action = random.choice(bestActions)
 
        return action


    def node_pacman_is_headed(self, state):
        pacman = state.getPacmanState()
        action = pacman.getDirection()
        px, py = state.getPacmanPosition()
        pacman_pos = (math.ceil(px), math.ceil(py))

        action_list = {
            "North" : (0,1),
            "South" : (0,-1),
            "East"  : (1,0),
            "West"  : (-1,0),
            "Stop"  : (0,0)
        }

        new_pacman_pos = (px + action_list[action][0], py + action_list[action][1])
        
        pacman_segment = self.get_segment(state.getPacmanPosition())    
        pacman_nodes = self.choice_dic[pacman_segment]

        for node in pacman_nodes:
            node_pos = self.segment_dic[node][0]
            if manhattanDistance(node_pos, new_pacman_pos) < manhattanDistance(node_pos, pacman_pos):
                self.last_node_pacman_was_heading_towards = node
                
        return self.last_node_pacman_was_heading_towards


    def node_pacman_is_moving_away_from(self, state):

        pacman_segment = self.get_segment(state.getPacmanPosition())    
        pacman_nodes = self.choice_dic[pacman_segment]

        if len(pacman_nodes) <=2:
            for node in pacman_nodes:
                if node != self.node_pacman_is_headed(state):
                    self.last_node_pacman_was_moving_away_from = node
        else:
            self.last_node_pacman_was_moving_away_from = pacman_segment
        
        return self.last_node_pacman_was_moving_away_from

 



