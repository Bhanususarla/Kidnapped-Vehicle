# Kidnapped-Vehicle

Your robot has been kidnapped and transported to a new location! 
Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data.

In this project we implemented a 2 dimensional particle filter in C++ to find the kidnapped robot. 
The particle filter is given a map and some initial localization information (analogous to what a GPS would provide). 
At each time step the filter also gets observation and control data.
