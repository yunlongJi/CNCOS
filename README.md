# A Cross-Network Node Classification Method in Open-Set Scenario


This repository contains code for the paper "A Cross-Network Node Classification Method in Open-Set Scenario".

Required packages:

•	tensorflow == 1.13.1


Datasets
===
data/ contains the 9 datasets used in our paper.

Each ".mat" file stores a network dataset, where

"network" represents an adjacency matrix, 

"attrb" represents a node attribute matrix,

"group" represents a node label matrix. 

And datasets in open-set scenario can be constructed based on the 9 datasets above.


File
===
"Blog_Test.py" is an example case of the cross-network node classification task from Blog1 to Blog2 networks.
