import math
import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from operator import itemgetter
from itertools import groupby
from networkx.algorithms import community
from numpy import loadtxt
from sklearn.metrics.cluster import normalized_mutual_info_score
import time

dataset_name = 'karate'
f = open('./datasets/'+dataset_name+'.txt', 'r')

graph = f.readlines()
N = len(graph)
start_time = time.time()
list_node_neighbors = np.array(np.zeros(N), dtype=object)
list_node_label = []
list_len_node = []
list_node_with_degree = []
for i in range(N):
    list_node_neighbors[i] = []
    list_node_label.append([i, 0])
    neis = str(graph[i]).split('\n')[0].split('\t')
    for j in neis:
        if j != '':
            list_node_neighbors[i].append(int(j))
    Len = len(list_node_neighbors[i])
    list_len_node.append(Len)
    list_node_with_degree.append((i, list_len_node[i]))
sort = sorted(list_node_with_degree, key=itemgetter(1))
group = groupby(sort, key=itemgetter(1))
low_high_degree = [tuple(x[0] for x in v) for k, v in group]
list_low_degree = []
if all(list_len_node[j] == 1 for j in low_high_degree[0]):
    list_low_degree.append(low_high_degree[0])
    low_high_degree.pop(low_high_degree.index(low_high_degree[0]))
counter_1 = 0
def highdegree(neighbor, label):
    list_L = []
    similarity = []
    degree = []
    for a in neighbor:
        L = list_node_label[a][1]
        CN = len(list(set(list_node_neighbors[label[0]]) & set(list_node_neighbors[a])))
        list_L.append(L)
        similarity.append((a, CN))
    count_label = Counter(list_L)
    if all(x == 0 for x in list_L):
        global counter_1
        counter_1 += 1
        label[1] = counter_1
        if similarity:
            max_value = max(similarity, key=itemgetter(1))[1]
            max_sim = [similarity[i][0] for i in range(len(similarity)) if similarity[i][1] == max_value]
            if len(max_sim) == 1:
                for i in max_sim:
                    list_node_label[i][1] = counter_1
            else:
                for i in max_sim:
                    d = list_len_node[i]
                    degree.append((i, d))
                max_node_degree = max(degree, key=itemgetter(1))[0]
                for j in [max_node_degree]:
                    list_node_label[j][1] = counter_1
    elif any(x != 0 for x in list_L):
        if all(j == 1 for j in count_label.values()):
            max_value = max(similarity, key=itemgetter(1))[1]
            max_sim = [similarity[i][0] for i in range(len(similarity)) if similarity[i][1] == max_value]
            if len(max_sim) == 1:
                for i in max_sim:
                    if list_node_label[i][1] == 0:
                        counter_1 += 1
                        label[1] = counter_1
                        list_node_label[i][1] = counter_1
                    else:
                        label[1] = list_node_label[i][1]
            else:
                for i in max_sim:
                    d = list_len_node[i]
                    degree.append((i, d))
                max_node_degree = max(degree, key=itemgetter(1))[0]
                for i in [max_node_degree]:
                    if list_node_label[i][1] == 0:
                        counter_1 += 1
                        label[1] = counter_1
                        list_node_label[i][1] = counter_1
                    else:
                        label[1] = list_node_label[i][1]
        elif any(j >= 2 for j in count_label.values()):
            max_count = max(count_label.values())
            max_count_label = [k for k, v in count_label.items() if v == max_count]
            if len(max_count_label) == 1:
                for i in max_count_label:
                    if i == 0:
                        node_withlabel = []
                        node_withzerolabel = []
                        for a in neighbor:
                            if list_node_label[a][1] != i:
                                d1 = list_len_node[a]
                                node_withlabel.append((a, d1))
                            elif list_node_label[a][1] == i:
                                d2 = list_len_node[a]
                                node_withzerolabel.append((a, d2))
                        max_node_withlabel = max(node_withlabel, key=itemgetter(1))[0]
                        max_node_withzerolabel = max(node_withzerolabel, key=itemgetter(1))[0]
                        for j in range(len(similarity)):
                            if similarity[j][0] == max_node_withlabel:
                                sim_node_withlabel = similarity[j][1]
                            elif similarity[j][0] == max_node_withzerolabel:
                                sim_node_withzerolabel = similarity[j][1]
                        if sim_node_withlabel >= sim_node_withzerolabel:
                            label[1] = list_node_label[max_node_withlabel][1]
                        else:
                            counter_1 += 1
                            label[1] = counter_1
                            list_node_label[max_node_withzerolabel][1] = counter_1
                    elif i != 0:
                        label[1] = i
            elif len(max_count_label) > 1:
                list_max_degree = []
                for i in max_count_label:
                    list_degree = []
                    for a in neighbor:
                        if list_node_label[a][1] == i:
                            d = list_len_node[a]
                            list_degree.append((a, d))
                    max_node_degree = max(list_degree, key=itemgetter(1))[0]
                    list_max_degree.append(max_node_degree)

                list_similarity = []
                for i in list_max_degree:
                    for j in range(len(similarity)):
                        if similarity[j][0] == i:
                            list_similarity.append(similarity[j])
                maxValue = max(list_similarity, key=itemgetter(1))[1]
                maxSimilarity = [list_similarity[j][0] for j in range(len(list_similarity)) if
                                 list_similarity[j][1] == maxValue]
                if len(maxSimilarity) == 1:
                    lst = []
                    for i in maxSimilarity:
                        if list_node_label[i][1] == 0:
                            for j in range(len(list_similarity)):
                                if list_similarity[j][0] != i:
                                    degree = list_len_node[list_similarity[j][0]]
                                    lst.append((list_similarity[j][0], degree))
                            max_node_degree = max(lst, key=itemgetter(1))[0]
                            for n in [max_node_degree]:
                                label[1] = list_node_label[n][1]
                        else:
                            label[1] = list_node_label[i][1]
                else:
                    node_Degree = []
                    for i in maxSimilarity:
                        d = list_len_node[i]
                        node_Degree.append((i, d))
                    max_node_degree = max(node_Degree, key=itemgetter(1))[0]
                    for j in [max_node_degree]:
                        if list_node_label[j][1] == 0:
                            lst_node = []
                            for k in range(len(node_Degree)):
                                if node_Degree[k][0] != j:
                                    lst_node.append(node_Degree[k])
                            max_degree = max(lst_node, key=itemgetter(1))[0]
                            for n in [max_degree]:
                                label[1] = list_node_label[n][1]

                        else:
                            label[1] = list_node_label[j][1]
for i in range(len(low_high_degree)):
    for j in low_high_degree[i]:
        if list_node_label[j][1] == 0:
            highdegree(list_node_neighbors[j], list_node_label[j])
list_degree_high_to_low = low_high_degree[::-1]
def Influence (node):
    neighborss = list(list_node_neighbors[node][0].keys())
    temp = []
    InfluenceScore_node_n = 0
    CN_sum = 0
    degreeSum = 0
    for n in neighborss:
        CNC = 0
        CommonNeighbors = (list(set(neighborss) & set(list_node_neighbors[n][0].keys())))
        L = len(CommonNeighbors)
        CN_sum += L
        list_node_neighbors[node][0].setdefault(n).append(L)
        for CN in CommonNeighbors:
            degreeSum += list_node_neighbors[CN][1]
            CNC += 1/list_node_neighbors[CN][1]
        exclusive_Node = 1
        exclusive_n = 1
        T = math.log(exclusive_Node + exclusive_n)
        NIC = CNC * T
        SECC = (L * list_node_neighbors[node][0][n][0]) / (math.sqrt(list_node_neighbors[node][1] * list_node_neighbors[n][1]))
        InfluenceScore_n_node = ((((L + 2)*NIC) / (list_node_neighbors[node][1] + 1))+list_node_neighbors[node][0][n][0]) + SECC
        InfluenceScore_node_n += (((L + 2)*NIC)/(list_node_neighbors[n][1] + 1))+list_node_neighbors[node][0][n][0] + SECC
        temp.append((InfluenceScore_n_node,n))
    if CN_sum == 0:
        NTS = list_node_neighbors[i][1]
    else:
        NTS = list_node_neighbors[i][1] + ((list_node_neighbors[i][1]*CN_sum)/degreeSum)
    max_inf = max(temp)
    list_node_neighbors.setdefault(node).append(node)
    list_node_neighbors.setdefault(node, []).append(max_inf[1])
    list_node_neighbors.setdefault(node, []).append(InfluenceScore_node_n+list_node_neighbors[node][1]+NTS)
    tempp = []
    tempp = ((list_node_neighbors[node][0][neighbor][1], neighbor) for neighbor in neighborss if (list_node_neighbors[neighbor][1] > 1))
    max_result = max(tempp, default=None)
    if (max_result is not None):
        list_node_neighbors.setdefault(node).append(1)
    else:
        list_node_neighbors.setdefault(node).append(0)
    list_node_neighbors.setdefault(node).append(0)
def neighborCommunityScore(node,neighborss):
    sum_scores = defaultdict(int)
    Weight_Similarity = defaultdict(float)
    moreThanTarget = defaultdict(float)
    for neighbor in neighborss:
        if list_node_neighbors[neighbor][1] > 1:
            neighbor_group = list_node_neighbors[neighbor][2]
            sum_scores[neighbor_group] += list_node_neighbors[neighbor][4]
            CommonNeighbors = list_node_neighbors[node][0][neighbor][1]*list_node_neighbors[node][0][neighbor][0]
            WeightedSimilarity = 2*CommonNeighbors / ((list_node_neighbors[node][1])+(list_node_neighbors[neighbor][1]))
            Weight_Similarity[neighbor_group] += WeightedSimilarity
            if list_node_neighbors[node][1] <= list_node_neighbors[neighbor][1]:
                moreThanTarget[neighbor_group] += 1
    return dict(sum_scores),dict(Weight_Similarity),moreThanTarget
def surrounding_communities(neighborss):
    communities = {}
    for ii in neighborss:
        if list_node_neighbors[ii][1] > 1:
            communities.setdefault(list_node_neighbors[ii][2], []).append(ii)
    return communities
def updateLabel(neighbors, nodeLabel):
    Label = []
    Similarity = []
    Degree = []
    for a in neighbors:
        node_label = list_node_label[a][1]
        sim = len(list(set(list_node_neighbors[nodeLabel[0]]) & set(list_node_neighbors[a])))
        Label.append(node_label)
        Similarity.append((a, sim))
    if Similarity:
        maxValue = max(Similarity, key=itemgetter(1))[1]
        maxSimilarity = [Similarity[i][0] for i in range(len(Similarity)) if Similarity[i][1] == maxValue]
        if len(maxSimilarity) == 1:
            for k in maxSimilarity:
                nodeLabel[1] = list_node_label[k][1]
        else:
            for j in maxSimilarity:
                d = list_len_node[j]
                Degree.append((j, d))
            maxDegree = max(Degree, key=itemgetter(1))[0]
            for i in [maxDegree]:
                nodeLabel[1] = list_node_label[i][1]
for i in range(len(list_degree_high_to_low)):
    for j in list_degree_high_to_low[i]:
        updateLabel(list_node_neighbors[j], list_node_label[j])
if len(list_low_degree) != 0:
    for i in list_low_degree[0]:
        for j in list_node_neighbors[i]:
            list_node_label[i][1] = list_node_label[j][1]
sorted_input = sorted(list_node_label, key=itemgetter(1))
groups = groupby(sorted_input, key=itemgetter(1))
communities = [tuple(x[0] for x in v) for k, v in groups]
def merge(Smallcomm, node_maxDegree, Bigcomm):
    listEdges = []
    list_node_with_maxdegree = []
    list_CommonNeighbor = []
    select_bigcommunity = []
    list_edge = []
    Node_Degree = []
    for j in Smallcomm:
        for k in Smallcomm:
            if j in list_node_neighbors[k]:
                if not listEdges:
                    listEdges.append((j, k))
                else:
                    if any(a[0] == k and a[1] == j for a in listEdges):
                        continue
                    else:
                        listEdges.append((j, k))
    inner_edge = len(listEdges)
    for k in range(len(Bigcomm)):
        new_list = []
        for v in Bigcomm[k]:
            degree = len(list_node_neighbors[v])
            new_list.append((v, degree))
        max_degree = max(new_list, key=itemgetter(1))[0]
        list_node_with_maxdegree.append(max_degree)
    for a in list_node_with_maxdegree:
        common_neighbor = len(list(set(list_node_neighbors[node_maxDegree]) & set(list_node_neighbors[a])))
        list_CommonNeighbor.append((a, common_neighbor))
    tashaboh = max(list_CommonNeighbor, key=itemgetter(1))[1]
    if tashaboh > 0:
        node_maxCommonneighbor = [list_CommonNeighbor[i][0] for i in range(len(list_CommonNeighbor)) if list_CommonNeighbor[i][1] == tashaboh]
        if len(node_maxCommonneighbor) == 1:
            for i in node_maxCommonneighbor:
                for j in Bigcomm:
                    if i in j:
                        select_bigcommunity.append(j)
            for x in Smallcomm:
                for j in range(len(select_bigcommunity)):
                    for y in select_bigcommunity[j]:
                        if x in list_node_neighbors[y]:
                            list_edge.append((x, y))
            outer_edge = len(list_edge)
            if (inner_edge / 2) - outer_edge <= 1:
                for j in Smallcomm:
                    list_node_label[j][1] = list_node_label[i][1]
        else:
            for z in node_maxCommonneighbor:
                deg = len(list_node_neighbors[z])
                Node_Degree.append((z, deg))
            Node_max = max(Node_Degree, key=itemgetter(1))[0]

            for c in Bigcomm:
                if Node_max in c:
                    select_bigcommunity.append(c)

            for x in Smallcomm:
                for j in range(len(select_bigcommunity)):
                    for y in select_bigcommunity[j]:
                        if x in list_node_neighbors[y]:
                            list_edge.append((x, y))
            outer_edge = len(list_edge)

            if (inner_edge / 2) - outer_edge <= 1:
                for j in Smallcomm:
                    list_node_label[j][1] = list_node_label[Node_max][1]
count_i = 0
def averageMerging(communities, count_i):
    global i
    len_community = []
    for i in communities:
        len_community.append(len(i))
    Avg = sum(len_community) / len(communities)
    list_Smallcommunity = []
    list_Bigcommunity = []
    for i in communities:
        if len(i) < Avg:
            list_Smallcommunity.append(i)
        else:
            list_Bigcommunity.append(i)
    list_maxdegree = []
    for k in range(len(list_Smallcommunity)):
        degree_node = []
        for n in list_Smallcommunity[k]:
            d = len(list_node_neighbors[n])
            degree_node.append((n, d))
        node = max(degree_node, key=itemgetter(1))[0]
        list_maxdegree.append(node)
    list_total = []
    for j in list_maxdegree:
        lst_1 = []
        neighbor = list_node_neighbors[j]
        for x in neighbor:
            for b in list_Bigcommunity:
                if x in b:
                    if b not in lst_1:
                        lst_1.append(b)
        list_total.append(lst_1)
    for com in range(len(list_Smallcommunity)):
        if len(list_total[com]) != 0:
            merge(list_Smallcommunity[com], list_maxdegree[com], list_total[com])
    sorted_input = sorted(list_node_label, key=itemgetter(1))
    groups = groupby(sorted_input, key=itemgetter(1))
    communities1 = [tuple(x[0] for x in v) for k, v in groups]
    count_i += 1
    if count_i == 1:
        averageMerging(communities1,count_i)
if dataset_name not in ["dblp", "amazon", "youtube"]:
    averageMerging(communities,count_i)
uniqueNodes = set()
ordered_nodes_neighbors = {}
for i in list_node_label:
    ordered_nodes_neighbors[i[0]] = i[1]
    uniqueNodes.add(i[1])
with open('./test-nodelabels.txt', 'w') as f:
    for item in range(len(list_node_label)):
        f.write("%s\n" % list_node_label[item][1])
#------------------LFR NMI---------------------------------------
# detected_labels = []
# for i in ordered_nodes_neighbors:
#     detected_labels.append(ordered_nodes_neighbors[i])
# detected_labels = np.array(detected_labels)
# real_labels= loadtxt("./LFR/"+numberOfNodes+"_"+mu+"/community.dat", comments="#", delimiter="\t", unpack=False)
# NMI = normalized_mutual_info_score(real_labels[0:len(real_labels),1],detected_labels)
# print(NMI)
# -------------------------------------------------------------
# def load_data(filename):
#     list_detected = list()
#     with open(filename) as file:
#         for line in file:
#             line = line.strip().split(',')
#             list_detected.append(int(line[0]))
#     return list_detected
# detected = load_data("./test-nodelabels.txt")
# copy_detected = detected.copy()
# ground_truth = load_data("./...")
# groundtruth_labels = []
# for i in range(len(ground_truth)):
#     groundtruth_labels.append((i, ground_truth[i]))
# sorted_label = sorted(groundtruth_labels, key=itemgetter(1))
# gp = groupby(sorted_label, key=itemgetter(1))
# ground_truth_community = [tuple(x[0] for x in v) for k, v in gp]
# for i in range(len(ground_truth_community)):
#     liSt = []
#     for n in ground_truth_community[i]:
#         liSt.append(copy_detected[n])
#     count = Counter(liSt)
#     if liSt:
#         X = max(count.values())
#         for k, v in count.items():
#             if v == X:
#                 label_with_maxDuplicate = k
#         for j in ground_truth_community[i]:
#             if copy_detected[j] == label_with_maxDuplicate:
#                 copy_detected[j] = groundtruth_labels[n][1]

print("--- Total Execution time %s seconds ---" % (time.time() - start_time))

# ----------------------------------- Modularity -----------------------------------------
print("Computing Modularity: ...")
t = 0
for i in range(N):

    t = t + len(list_node_neighbors[i])

edges = t / 2

modu = 0
are_neighbor = []

for i in range(N):

    neighborsList = list_node_neighbors[i]
    for j in ordered_nodes_neighbors.keys():

        if ordered_nodes_neighbors[i] == ordered_nodes_neighbors[j]:

            if len(list_node_neighbors[i]) >= 1:

                if j in neighborsList:

                    are_neighbor = 1

                else:

                    are_neighbor = 0

                modu = modu + (are_neighbor - ((len(list_node_neighbors[i]) * len(list_node_neighbors[j])) / (2 * edges)))

modularity_final = modu / (2 * edges)

print('Modularity:  {}'.format(modularity_final))

#-------------------------------- NMI----------------------------------

real_labels =  loadtxt("./groundtruth/"+dataset_name+"_real_labels.txt", comments="#", delimiter="\t",unpack=False)
detected_labels = loadtxt("./test-nodelabels.txt", comments="#",delimiter=",", unpack=False)
nmi = normalized_mutual_info_score(real_labels, detected_labels)
print("nmi:", nmi)

#---------------- nodes map for big dataset------------------------
# detected_labels = []
# nodes_map = loadtxt("datasets/nodes_map/" + dataset_name + "_nodes_map.txt", comments="#", delimiter="\t", unpack=False)
# for i in nodes_map:
#     detected_labels.append(ordered_nodes_neighbors[i])
# print('NMI:  {}'.format(normalized_mutual_info_score(real_labels, detected_labels)))
