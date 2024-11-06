# Mall-customer-segmentation

requirements and packages:
numpy
panda
matplotlib
seaborn
sklearn

Step1: we need to decide the number of custers the customers can be segmented into. For that, we calculate wccb for each number of clusters from 1 cluster to 11 clusters.
Step2 : Plot the wccb vs number of clusters graph to observe the sharpest point on the plot. The number of clusters corresponding to this point will be the number of clusters to take, which means customers are to be segmented into 4 groups.
Step3 : Training the K-Means clustering model.
Step4 : Plotting the segmented data.
