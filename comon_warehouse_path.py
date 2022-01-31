from Environment_maze_2 import Maze
from Agent import QLearningTable, ReturnQLearningTable
import matplotlib.pyplot as plt
import pickle
HUMANWALK1 = [1,1,1,1,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
UNIT = 20
def update():
    totalReward1 = 0
    totalReward2 = 0
    totalReward3 = 0
    rewardList1 = []
    rewardList2 = []
    rewardList3 = []
    totalRewardList = []
    freeze1 = False
    freeze2 = False
    freeze3 = False
    #for fixed obstacal
    backup1 = False
    backup2 = False
    backup3 = False
    
    for episode in range(500):
        # initial observation
        observation1, observation2, observation3 = env.resetRobot()
        human1, human2 = env.resetHuman()
        humanWalkHelper = 0
        wait_time1, wait_time2, wait_time3 = 0,0,0
        
        while True:
            # fresh env
            env.render()

            if humanWalkHelper < len(HUMANWALK1):
                env.humanStep1(HUMANWALK1[humanWalkHelper])
            else:
                env.humanStep1(4)
            humanWalkHelper +=1

            
            # RL choose action based on observation
            if freeze1:
                action1 = 4
            else:
                if not backup1:
                    action1 = chooseAction(episode, RL1, observation1)   
                    if stateChecking(human1, observation1, action1) == 'no_collision':
                        observation1_, reward1, done1 = env.step1(action1)
                    else:
                        if wait_time1 > 5:
                            if stateChecking(human1, observation1, action1) == 'upward_collision':
                                observation1_, reward1, done1 = env.step1(1)
                            elif stateChecking(human1, observation1, action1) == 'downward_collision':
                                observation1_, reward1, done1 = env.step1(0)
                            elif stateChecking(human1, observation1, action1) == 'left_collision':
                                observation1_, reward1, done1 = env.step1(2)
                            elif stateChecking(human1, observation1, action1) == 'right_collision':
                                observation1_, reward1, done1 = env.step1(3)
                            backup1 = True
                            wait_time1 = 0
                        else:
                            observation1_, reward1, done1 = env.step1(4)
                            wait_time1 +=1 
                    learn (episode, RL1, action1, reward1, observation1, observation1_)
                else:
                    action1 = chooseAction(episode, backupRL1, observation1)
                    observation1_, reward1, done1 = env.step1(action1, human1)
                    learn (episode, backupRL1, action1, reward1, observation1, observation1_)
                totalReward1+=reward1
                observation1 = observation1_
            
            if freeze2:
                action2 = 4
            else:                
                if not backup2:
                    action2 = chooseAction(episode, RL2, observation2)   
                    if stateChecking(human1, observation2, action2) == 'no_collision':
                        observation2_, reward2, done2 = env.step2(action2)
                    elif stateChecking(RL1, observation2, action2) == 'no_collision':
                        observation2_, reward2, done2 = env.step2(action2)
                    else:
                        if wait_time2 > 5:
                            if stateChecking(human1, observation2, action2) == 'upward_collision' or stateChecking(RL1, observation2, action2) == 'upward_collision':
                                observation2_, reward2, done2 = env.step2(1)
                            elif stateChecking(human1, observation2, action2) == 'downward_collision' or stateChecking(RL1, observation2, action2) == 'downward_collision':
                                observation2_, reward2, done2 = env.step2(0)
                            elif stateChecking(human1, observation2, action2) == 'left_collision' or stateChecking(RL1, observation2, action2) == 'left_collision':
                                observation2_, reward2, done2 = env.step2(2)
                            elif stateChecking(human1, observation2, action2) == 'right_collision' or stateChecking(RL1, observation2, action2) == 'right_collision':
                                observation2_, reward2, done2 = env.step2(3)
                            backup2 = True
                            wait_time2 = 0
                        else:
                            observation2_, reward2, done2 = env.step2(4)
                            wait_time2 +=1 
                        learn (episode, RL2, action2, reward2, observation2, observation2_)
                else:
                    action2 = chooseAction(episode, backupRL2, observation2)
                    observation2_, reward2, done2 = env.step2(action2, human1)
                    learn (episode, backupRL2, action2, reward2, observation2, observation2_)
                totalReward2+=reward2
                observation2 = observation2_
            
            if freeze3:
                action3 = 4
            else:
                if not backup3:
                    action3 = chooseAction(episode, RL3, observation3)   
                    if stateChecking(human1, observation3, action3) == 'no_collision':
                        observation3_, reward3, done3 = env.step3(action3)
                    else:
                        if wait_time3 > 5:
                            if stateChecking(human1, observation3, action3) == 'upward_collision':
                                observation3_, reward3, done3 = env.step3(1)
                            elif stateChecking(human3, observation3, action3) == 'downward_collision':
                                observation3_, reward3, done3 = env.step3(0)
                            elif stateChecking(human1, observation3, action3) == 'left_collision':
                                observation3_, reward3, done3 = env.step3(2)
                            elif stateChecking(human3, observation3, action3) == 'right_collision':
                                observation3_, reward3, done3 = env.step3(3)
                            backup3 = True
                            wait_time3 = 0
                        else:
                            observation3_, reward3, done3 = env.step3(4)
                            wait_time3 +=1 
                    learn (episode, RL3, action3, reward3, observation3, observation3_)
                else:
                    action3 = chooseAction(episode, backupRL3, observation3)
                    observation3_, reward3, done3 = env.step3(action3, human1)
                    learn (episode, backupRL3, action3, reward3, observation3, observation3_)
                totalReward3+=reward3
                observation3 = observation3_
            
            # break while loop when end of this episode

            if (done1 == 'hit' or done1 == 'arrive') and (done2 == 'hit' or done2 == 'arrive') and (done3 == 'hit' or done3 == 'arrive'):
                print (episode, 'trial: ','Robot1: ', totalReward1, '; Robot2: ', totalReward2, '; Robot3: ', totalReward3)
                #print (freeze1, freeze2, freeze3)
                rewardList1.append(totalReward1)
                rewardList2.append(totalReward2)
                rewardList3.append(totalReward3)
                totalRewardList.append(totalReward1+totalReward2+totalReward3)
                totalReward1 = 0
                totalReward2 = 0
                totalReward3 = 0
                if done1 == 'arrive' and done2 == 'arrive' and done3 == 'arrive':
                    for i in range(50):
                        if startReturnTable (episode, observation1, ReturnRL1,1) != 'nothing':
                            break
                    for i in range(50): 
                        if startReturnTable (episode, observation2, ReturnRL2,2) != 'nothing':
                            break
                    for i in range(50): 
                        if startReturnTable (episode, observation3, ReturnRL3,3) != 'nothing': 
                            break
                            
                freeze1 = False
                freeze2 = False
                freeze3 = False
                break
            if done1 == 'hit' or done1 == 'arrive':
                freeze1 = True
            if done2 == 'hit' or done2 == 'arrive':
                freeze2 = True
            if done3 == 'hit' or done3 == 'arrive':
                freeze3 = True

    '''
    f1 = open('Return_qtable1', 'wb')
    pickle.dump(ReturnRL1.q_table,f1)
    f1.close()
    f2 = open('Return_qtable2', 'wb')
    pickle.dump(ReturnRL2.q_table,f2)
    f2.close()
    f3 = open('Return_qtable3', 'wb')
    pickle.dump(ReturnRL3.q_table,f3)
    f3.close()
    f4 = open('path_qtable1', 'wb')
    pickle.dump(RL1.q_table,f4)
    f4.close()
    f5 = open('path_qtable2', 'wb')
    pickle.dump(RL2.q_table,f5)
    f5.close()
    f6 = open('path_qtable3', 'wb')
    pickle.dump(RL3.q_table,f6)
    f6.close()
    '''
    plot(rewardList1,"agent - 1")
    plot(rewardList2,"agent - 2")
    plot(rewardList3,"agent - 3")   
    plot(totalRewardList,"Sum of Total Reward")                          
    print('game over')
    env.destroy()   


def chooseAction (episode, RL, observation):
    if episode < 100:
        return RL.choose_action(str(observation), 0.9+episode*0.001)
    else:
        return RL.choose_action(str(observation),1)
def learn (episode, RL, action, reward, observation, observation_):
     if episode < 500:
         RL.learn(str(observation), action, reward, str(observation_), 0.03, 0.9)
     elif episode < 1500 and episode >= 500:
         RL.learn(str(observation), action, reward, str(observation_), 0.3-0.0002*(episode-500), 0.9)
     else:
         RL.learn(str(observation), action, reward, str(observation_), 0.001, 0.9)
    
def startReturnTable (episode, observation, RL, robotNumber):
    while True:
        env.render()
        action = chooseNoRandomAction(RL, observation)
        if robotNumber == 1:
            observation_, reward, done = env.returnStep1(action)
        elif robotNumber == 2:
            observation_, reward, done = env.returnStep2(action)
        elif robotNumber == 3:
            observation_, reward, done = env.returnStep3(action)
        learn (episode, RL, action, reward, observation, observation_)
        observation = observation_
        if done == 'arrive' or done == 'hit':
            break
    return done

def chooseNoRandomAction(RL, observation):
    return RL.choose_action(str(observation),1)


def plot (reward, title):
    plt.style.use('seaborn-deep')
    plt.plot(reward,linewidth= 0.7, color='red')
    plt.title(title)
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.show() 
def stateChecking(alien_agent, key_agent, action):
    directEnvironment = directNearbyEnvironment(key_agent)
    indirectEnvironment = indirectNearbyEnvironment(key_agent)
    if action == 0 and (alien_agent in [directEnvironment[0], indirectEnvironment[0], indirectEnvironment[1]]):
            return 'upward_collision'
    elif action == 1 and (alien_agent in [directEnvironment[1], indirectEnvironment[2], indirectEnvironment[3]]):
            return 'downward_collision'
    elif action == 2 and (alien_agent in [directEnvironment[3], indirectEnvironment[1], indirectEnvironment[3]]):
            return 'right_collision'
    elif action == 3 and (alien_agent in [directEnvironment[2], indirectEnvironment[0], indirectEnvironment[2]]):
            return 'left_collision'
    else:
        return 'no_collision'

def indirectNearbyEnvironment(coordinate):
    upleft = [coordinate[0]-UNIT, coordinate[1]-UNIT, coordinate[2]-UNIT, coordinate[3]-UNIT]
    upright = [coordinate[0]+UNIT, coordinate[1]-UNIT, coordinate[2]+UNIT, coordinate[3]-UNIT]
    downleft = [coordinate[0]-UNIT, coordinate[1]+UNIT, coordinate[2]-UNIT, coordinate[3]+UNIT]
    downright = [coordinate[0]+UNIT, coordinate[1]+UNIT, coordinate[2]+UNIT, coordinate[3]+UNIT]
    nearby = [upleft, upright, downleft, downright]
    return nearby   


def directNearbyEnvironment(coordinate):
    left = [coordinate[0]-UNIT, coordinate[1], coordinate[2]-UNIT, coordinate[3]]
    right = [coordinate[0]+UNIT, coordinate[1], coordinate[2]+UNIT, coordinate[3]]
    up = [coordinate[0], coordinate[1]-UNIT, coordinate[2], coordinate[3]-UNIT]
    down = [coordinate[0], coordinate[1]+UNIT, coordinate[2], coordinate[3]+UNIT]
    nearby = [up, down, left, right]
    return nearby

if __name__ == "__main__":
    env = Maze()
    RL1 = QLearningTable(actions=list(range(env.n_actions)), path='Trained_data/path_qtable1')
    RL2 = QLearningTable(actions=list(range(env.n_actions)), path='Trained_data/path_qtable2')
    RL3 = QLearningTable(actions=list(range(env.n_actions)), path='Trained_data/path_qtable3')
    backupRL1 = QLearningTable(actions=list(range(env.n_actions)),path='Trained_data/q_table1')
    backupRL2 = QLearningTable(actions=list(range(env.n_actions)),path='Trained_data/q_table2')
    backupRL3 = QLearningTable(actions=list(range(env.n_actions)),path='Trained_data/q_table3')
    ReturnRL1 = ReturnQLearningTable(actions=list(range(env.n_actions)),path='Trained_data/Return_qtable1')
    ReturnRL2 = ReturnQLearningTable(actions=list(range(env.n_actions)),path='Trained_data/Return_qtable1')
    ReturnRL3 = ReturnQLearningTable(actions=list(range(env.n_actions)),path='Trained_data/Return_qtable1')
    env.after(3000, update)
    env.mainloop()
