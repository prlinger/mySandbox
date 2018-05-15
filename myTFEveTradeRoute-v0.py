# This is the first version of a RL algorithm that takes eve online
# as the environment and finds a path between two star systems,
# avoiding higher risk space (low sec) and minimizing distance.
# To start I'll focus on pathfinding in arbitrary graphs (not imported from Eve).
# After that works, there are a many extensions that I can work on.
# - Using the eve environment.
# - Getting market buy and sell orders and using these to choose where to trade
#       then finding a good route.
# - Stringing together multiple trades.
#   - can carry extra items with.
#   - Also leaving from the state left from the last trade.
# - Changing behaviour based off of system ship kill statistics from recent hours.
#   - eg. A high number of deaths indicate a gate camp.  Avoid that system.
# Notes:
# lowsec denotes low security space in Eve.  ie. More dangerous  

import numpy as np
import pylab as plt
import networkx as nx
import tensorflow as tf
import tensorflow.contrib.slim as slim

# Create the environment.
points_list = [(0,1), (0,2), (1,3), (1,4), (2,3), (2,4), (3,5), (4,5)]
goal = 5
# lowsec denotes low security star systems in Eve.  ie. More dangerous
lowsecSystems = [2, 4]
# highsec is safer space.
highsecSystems = [0, 1, 3, 5]

def showGraph():
    G = nx.Graph()
    G.add_edges_from(points_list)
    mapping = {0:'Start', 5:'End'}
    G = nx.relabel_nodes(G,mapping)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=["g","g","r","g","r","g"])
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()




class Agent():
    """Our agent."""
    def __init__(self):
        print("test")
        # establish feedforward part of network
        # establish training procedure
        



########################
# Training:
########################

# tf.reset_default_graph() # clear the tensorflow graph
















