import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate
import collections

# Import karate club data
zachary_data = nx.karate_club_graph()


# Network diagram 
def draw_node_size_by_deg (g, scale=None, with_labels=True): 
    plt.figure(figsize=(5,4))
    degree = nx.degree(g) 
    node_list = [n for (n,m) in degree]
    degree_list = [int(m) for (n, m) in degree]  
    nx.draw_circular(g, with_labels=with_labels,nodelist=node_list)
    plt.title('Circular Layout')
    plt.show()

print(draw_node_size_by_deg(zachary_data, with_labels=True))
print (nx.info(zachary_data))
print ("Network diameter: ", nx.diameter(zachary_data))


# degree_distribution plot
def plot_degree_distribution(g):

    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots(figsize=(5,4))
    plt.bar(deg, cnt, width=0.80, color="b")
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    plt.show()

plot_degree_distribution(zachary_data)  


def print_clustering_coefficient(g):
    data_c = nx.clustering(g)
    data ={'Clustering Coefficient':[]}
    for i in range(34):
        data['Clustering Coefficient'].append(data_c[i])
    print(tabulate(data, headers='keys',showindex=True))
    
print_clustering_coefficient(zachary_data)


# Network diagram 
def draw_node_size_by_deg (g, scale=None, with_labels=True): 
    plt.figure(figsize=(12,7))
    degree = nx.degree(g) 
    node_list = [n for (n,m) in degree]
    degree_list = [int(m) for (n, m) in degree]  
    plt.subplot(1,2,1)
    nx.draw_kamada_kawai(g, with_labels=with_labels,
                         nodelist  =node_list,
                         node_size =[n*100 for n in degree_list])
    plt.title('Kamada-Kawai algorithm Layout')
    plt.show()

draw_node_size_by_deg(zachary_data, with_labels=True)


# Closeness centrality of the verticeser
def closeness_cent(g):
    closeness_centrality_dict = nx.closeness_centrality(zachary_data) # closeness centrality for each node
    closeness = closeness_centrality_dict.items()
    for clos in closeness: 
            print(clos) 
print('============ Closeness centrality of the vertices==============')
print('')
closeness_cent(zachary_data)


# Betweenness centrality of the vertices
def betweenness_cent (g): 
    betweenness_centrality_dict = nx.betweenness_centrality(zachary_data) # calculate betweenness centrality for each node
    betweenness = betweenness_centrality_dict.items()
    for bwt in betweenness:
        print(bwt) 

print('============ Betweenness centrality of the vertices==============')
print('')
betweenness_cent(zachary_data)

