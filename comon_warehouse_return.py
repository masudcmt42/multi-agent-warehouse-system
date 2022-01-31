from Environment_maze import Maze
from Agent import QLearningTable, ReturnQLearningTable
import matplotlib.pyplot as plt
import pickle
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
    for episode in range(3000):
        # initial observation
        observation1, observation2, observation3 = env.resetRobot()
        
        while True:
            # fresh env
            env.render()
            
            # RL choose action based on observation
            if freeze1:
                action1 = 4
            else:
                # Choose action
                action1 = chooseAction(episode, RL1, observation1)   
                # RL take action and get next observation and reward                
                observation1_, reward1, done1 = env.step1(action1)
                totalReward1+=reward1
                # RL learn from this transition
                learn (episode, RL1, action1, reward1, observation1, observation1_)
                # swap observation
                observation1 = observation1_
            
            if freeze2:
                action2 = 4
            else:                
                # Choose action
                action2 = chooseAction(episode, RL2, observation2)        
                # RL take action and get next observation and reward                
                observation2_, reward2, done2 = env.step2(action2)
                totalReward2+=reward2
                # RL learn from this transition
                learn (episode, RL2, action2, reward2, observation2, observation2_)
                # swap observation
                observation2 = observation2_
            
            if freeze3:
                action3 = 4
            else:
                # Choose action
                action3 = chooseAction(episode, RL3, observation3) 
                # RL take action and get next observation and reward         
                observation3_, reward3, done3 = env.step3(action3)
                totalReward3+=reward3
                # RL learn from this transition
                learn (episode, RL3, action3, reward3, observation3, observation3_)
                # swap observation
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
                        if startReturnTable (episode, observation1, ReturnRL1,1) == 'arrive':
                            break
                    for i in range(50): 
                        if startReturnTable (episode, observation2, ReturnRL2,2) == 'arrive':
                            break
                    for i in range(50): 
                        if startReturnTable (episode, observation3, ReturnRL3,3) == 'arrive': 
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



        if episode == 2500:     
            f1 = open('Trained_data/RQ_table1', 'wb')
            pickle.dump(RL1.q_table,f1)
            f1.close()
            f2 = open('Trained_data/RQ_table2', 'wb')
            pickle.dump(RL2.q_table,f2)
            f2.close()
            f3 = open('Trained_data/RQ_table3', 'wb')
            pickle.dump(RL3.q_table,f3)
            f3.close()
        '''
        if episode == 2500:     
            f1 = open('Trained_data/q_table1', 'wb')
            pickle.dump(RL1.q_table,f1)
            f1.close()
            f2 = open('Trained_data/q_table2', 'wb')
            pickle.dump(RL2.q_table,f2)
            f2.close()
            f3 = open('Trained_data/q_table3', 'wb')
            pickle.dump(RL3.q_table,f3)
            f3.close()   
            
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
    #observation1, observation2, observation3 = env.resetRobot()
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
        #print (done)
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
    
if __name__ == "__main__":
    env = Maze()
    RL1 = QLearningTable(actions=list(range(env.n_actions)), path='Trained_data/q_table1')
    RL2 = QLearningTable(actions=list(range(env.n_actions)), path='Trained_data/q_table2')
    RL3 = QLearningTable(actions=list(range(env.n_actions)), path='Trained_data/q_table3')
    ReturnRL1 = ReturnQLearningTable(actions=list(range(env.n_actions)))
    ReturnRL2 = ReturnQLearningTable(actions=list(range(env.n_actions)))
    ReturnRL3 = ReturnQLearningTable(actions=list(range(env.n_actions)))
    env.after(3000, update)
    env.mainloop()
