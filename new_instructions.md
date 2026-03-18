1. Establish the same standard for the network metrics recording object (Pandas dataframe).
In the last set of codes, we had a disparity in some graphics, even when the calculations done by you were correct. Those disparities  were caused by the difference in object selection (dictionary, list, dataframe). In order to avoid it, please lets covert everything to the same object (dataframe) and stick to it. The first two columns of the dataframe should be named "i" and "j", and they will represent the nodes on the existing edges. Each metric (LDC, Jaccard, etc.) should be added as an additional column. So as a result, each row will represent the edge between the pair of "i" and "j" nodes, and all the values that correspond to the (LDC, Jaccard, LKS, LLBCe, LLBMEe, and CI).
The reason for selecting this format is that most datasets are stored as CSV or Excel files, which are naturally read into Python as dataframes.

2. Please, redo the CI method, from https://www.mdpi.com/1099-4300/26/3/248. Your solutions were incorrect, not due to your mistake in coding, but due to a bad method explanation in the text. In the paper that we checked, it is written "∂Ball⁡(𝑢,𝓁) denotes the set of nodes in the network whose shortest path length to u is ℓ. Here, we set ℓ to be 3." But in the original work, it is introduced as "Ball(i, 𝓁) is the set of nodes inside a ball of radius 𝓁 (defined as the shortest path) around node i." Let's consider the 𝓁=3 as it is in the papers.
In other words, it is a "skin" of the ball or the "whole body with volume" of the ball.
I am asking you to calculate CI using both assumptions. CI_skin, and CI_body. CI_skin is described in the paper (https://www.mdpi.com/1099-4300/26/3/248), but it is questionable due to its very specific focus on the 𝓁_s order neighbors, when CI_body represents all the neighbors up to the 𝓁_s order.

Also, after you calculate the CI_sking and CI_body, you will face a challenge in deciding how to describe the edge, since the metric is assigned to a node. For the current moment of time, lets calculate it in two ways: average and multiplication.
CI_e_av=(CI_i+CI_j)/2; CI_e_mul=CI_i*CI_j
So as a result you will have to obtain 4 columns out of CI.


3. Please, drop the EI method (from the same article https://www.mdpi.com/1099-4300/26/3/248). The EI method has multiple flaws in description, and there is no chance of being certain about what the authors mean.

4. And the last, please do another two methods, LLBC𝑒 and LLBME𝑒1, from the paper https://www.mdpi.com/1099-4300/26/4/315. Those are the methods described by equations 8 and 11. The key point is that you must use the function "nx.edge_betweenness_centrality_subset". Cutting subgroups from the graph, or constructing subgraphs via neighbors, will give you the wrong output.



From here, you will have a set of "static" algorithms and decomposed graphs according to them: LDC, Jaccard, LKS, LLBCe, LLBMEe, CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body. After confirming that all of those are calculated correctly we can move to iteration of those.