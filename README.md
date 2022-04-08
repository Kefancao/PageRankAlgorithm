# Page Rank Algorithm

## Introduction
The page rank algorithm is used by search engines to order the weblinks of a particular search result in a way that is most desirable for someone looking to find more information on the topic. It works by determining which webpage has the most links pointed to it. For example, if ABC.com is a useful website containing desirable information when people search "ABC", then naturally, other websites when referring to the topic of "ABC" would quote ABC.com, and therefore creating a link to it. The more and more websites think ABC.com is the go to site for topics on ABC, the more important and higher up the algorithm will think it is, thus ranking it higher up. This is essentially how the page rank algorithm works. 

The python file in this repo is meant to act as a demonstration to give the ranking for the very small internet graph that follows. 

## Graphs
The entirety of the internet can be thought of as a graph of nodes and edges. Each outgoing edge from a node can be viewed as a vote on the importance of the node on which this edge points to. Through this, we can construct a transition matrix to simulate the behaviour and derive a rank on the order of importance of all these pages. The python script in this repo attempts to do this for the following mini internet graph. 

<img width="491" alt="Screen Shot 2022-03-30 at 1 22 25 PM" src="https://user-images.githubusercontent.com/76069770/160894535-36777111-a60e-4d16-bc33-b0d05fb2115c.png">

Of course, there are many details this explanation dismisses, such as the importance of a vote from a node, the likelihood of deadend pages, convergence of the transition matrix, etc. For more information on this topic, this wikipedia page provides a great starting point [link:"https://en.wikipedia.org/wiki/PageRank"]
