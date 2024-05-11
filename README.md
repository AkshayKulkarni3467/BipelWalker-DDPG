The deep deterministic policy gradient (DDPG) algorithm is a model-free, online, off-policy reinforcement learning method. A DDPG agent is an actor-critic reinforcement learning agent that searches for an 
optimal policy that maximizes the expected cumulative long-term reward. 

DDPG relies on external noise for exploration and converges well for actions which are not predicted by the deterministic policy. For example:

![LunarLanderDDPG](https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/5a92b0bf-8aa2-4f9e-9f38-16e927b311e0)

This make the agent take various random actions which maximize the reward at that time step. For example, in bipedal walker, some of the random actions are: 


https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/821dd02e-6cc8-4740-9e95-0f9de95631d0


The final stages of learning of Bipedal Walker:


https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/508757ef-fbd4-4247-a6ef-6c178c690252


The final trained example:


![BipedTrained](https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/3fa2bcc5-23f0-404f-8d50-674f8cfdb224)


The policy loss curve, Q-loss curve and mean reward curve:


![policy loss](https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/5170bac0-c2eb-4d3b-89cd-ae38ce6fcdd6)


![Q loss](https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/b22a2d39-5b8c-492d-859f-bc575e5c4a29)


![mean reward](https://github.com/AkshayKulkarni3467/BipelWalker-DDPG/assets/129979542/3708a407-bdac-4d60-836d-8a8cbe8b485c)


