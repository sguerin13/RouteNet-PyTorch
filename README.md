# PyTorch Implementation of [RouteNet](https://github.com/happma/RouteNet-challenge)
 - RouteNet models a computer network as a series of links and paths, where the model is tasked with predicting per path delay and jitter. 
 - The model has a multi-step message passing routine, first aggregating state information for the links in a given path, and then passing the sequence of link states through a RNN to update the path state.
 - The updated path states are then used to update the link states for all paths that utilize a given link.
 - This implementation was done as part of a group project for CSE 222 at UC San Diego, where I was responsible for the model creation and training.
   - This repo is a simplified and lightly refactored version of the original project [Repo](https://github.com/sguerin13/CSE222RouteNet).
 
