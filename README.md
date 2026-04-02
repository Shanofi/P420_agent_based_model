Introduction:
---------------------------------------------------------------------------------------------------------------------------------------------

We are a group of students from the University of Waterloo currently enrolled in Psychology 420: An Introduction to Computational Neuroscience Methods.

As part of our coursework focusing on Agent-Based Modeling (ABM), we have completed our final project. Please find our presentation slides, source code, and final research paper attached for your review.

In Regards to the Segregation model
---------------------------------------------------------------------------------------------------------------------------------------------

In this model, we simulate a 2D space where the agents can live in different locations. Agents can also decide whether they want to move or stay at their location based on the similarity of their neighbors. Agents are happier if they live near other agents with similar characteristics and less happy when they are surrounded by agent(s) of a different characteristic.

Once an agent becomes unhappy, the agent moves to a new location chosen at random from among all vacant locations. At the end of each time period, we measure the number of agents that would like to move (the "unhappy" count) as well as the overall level of satisfaction (or "happiness") in the entire system.

We continue to run the simulation until no more agents wish to move, indicating that the system has reached equilibrium.

This dynamic behavior allows us to see how a simple local rule used by individuals in the system leads to complex global outcomes. In particular, we find that small differences in preferences lead to significant variations in the distribution of agents across locations.

Key Features
---------------------------------------------------------------------------------------------------------------------------------------------

• Interactive 2-D Grid
• Real-Time Charts 
• User-Defined Parameters 
• Two Modes of Operation

User Interface
---------------------------------------------------------------------------------------------------------------------------------------------

Below is a list of controls available within the user interface:

• INIT Button: Resets the grid using current values for parameters. This means that the initial conditions for both the grid configuration and parameter values are recreated each time INIT is pressed. It is assumed that users intend to test various combinations of parameters.

• STEP Button: Advances one iteration of the simulation. Each time STEP is clicked, the number of unhappy agents and the average happiness score increase and/or decrease depending upon whether there are enough similar neighbors present. If there aren't sufficient numbers of similarly colored neighbors nearby then those unhappy agents are moved to vacant homes. The STEP button is useful for debugging purposes or to explore intermediate states. Users can advance as few or as many iterations as desired; however, they cannot observe changes during an ongoing simulation.

• RUN Button: Starts the automated simulation process. When RUN is selected, the simulation proceeds continuously and automatically updates the display at regular intervals. Pressing STOP will halt further progress until RUN is once again activated. If RUN is clicked while already running, the next frame will be displayed immediately after clicking STOP.

• STOP Button: Stops automatic updating of the graph when RUN is active.

• PAUSE Button: Pauses but doesn't stop simulations completely. When PAUSE is pressed, users can make additional selections or modify parameters without losing their position in the simulation.

• SLIDERS: Three adjustable sliders allow users to define three types of input variables: grid size, threshold value and percent vacant (vacancy percentage).

Requirements
---------------------------------------------------------------------------------------------------------------------------------------------

Python version 3.8+

numpy
matplotlib
scipy
