

""" A program to implement a reinforcement learning algorithm, utilising a simple bandit algorithm and improve upon it by incorporating a Q function for the Golf Scenario
"""

import time
from tkinter import *
from PIL import ImageTk, Image
from random import randint
import numpy as np
import math
import sys

# Start global variables #
quit = False
width = 1020
height = 558
node_size = 10
delay = 0.05
inter = 0

#golf clubs
driver = 3
iron = 2
putter = 1

golf_clubs = [driver, iron, putter]
terminal_states = [6,7]

# node locations:
node_locs = [[56,319],[198,303],[324,264],[452,218],[584,194],[699,264] ,[764,298], [930,312]]
rewards = [-6,-5,-4,-3,-2,-1,0,-6]

# Q learning variables
iterations = 1000
# Learning rate was changed from 0.9 to 0.1 to find the optimal solution
learning_rate = 0.1
gamma = 0.9
tests = len(node_locs)
k = len(golf_clubs)


visited = []
node_items = []

def create_node(x, y, r, canvasName,colour,obj_tag):
    """Function create_node to create a node

        Arguments:
            x -- The x coordinate
            y -- The y coordinate
            r -- The radius of the node
            canvasName -- The canvas to draw the node
            colour -- The colour of the node
            obj_tag -- The tag of the node

        Returns:
            canvasName.create_oval(x0, y0, x1, y1,fill=colour,tag=obj_tag)
    """
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1,fill=colour,tag=obj_tag)

def create_branch(x0, y0, x1, y1, canvasName,colour,obj_tag):
    """Function create_branch to create a branch

        Arguments:
            x0 -- The x0 coordinate
            y0 -- The y0 coordinate
            x1 -- The x1 coordinate
            y1 -- The y1 coordinate
            canvasName -- The canvas to draw the branch
            colour -- The colour of the branch
            obj_tag -- The tag of the branch

        Returns:
            None

    """

    "Create line with fill colour: (x0,y0 to x1,y1): "
    return canvasName.create_line(x0, y0, x1, y1,fill=colour,width=3,tag=obj_tag)



def create_plot(myCanvas):
    """Function create_plot to create the plot

        Arguments:
            myCanvas -- The canvas to draw the plot

        Returns:
            None

    """
    global node_size
    global node_locs
    global visited
    global node_items

    #delete before redraw
    for item in node_items:
       myCanvas.delete(item)

    node_label = 1
    #draw/redraw nodes
    for x,y in node_locs:
        node_name = "n"+str(x)+str(y)
        if node_name in visited:
            pass #create_node(x,y,node_size,myCanvas,"darkred",node_name)
        else:
            create_node(x,y,node_size,myCanvas,"red",node_name)
        text_name = "t"+str(node_label)
        myCanvas.create_text(x, y, text=str(node_label),tag=text_name, font=(20) )
        node_label = node_label + 1
        if node_name not in node_items:
            node_items.append(node_name)
            node_items.append(node_label)

def get_Q_path(Q):
    """Function get_Q_path based on the Q function

        Arguments:
            Q -- The Q function matrix

        Returns:
            reward -- The reward
    """
    global visited
    reward = 0
    hole = 0
    tests = len(rewards)-len(terminal_states) #tests every state
    while(True):
        club = np.argmax(Q[hole])
        swing = 0 # randint(-1,1)
        hole += golf_clubs[club] + swing

        if hole > 7:
           hole = 7
           reward += -1
        visited.append(hole)
        reward += rewards[hole]
        if hole in terminal_states:
           break
    return reward

def Q_function():
    """Function Q_function  implements the Q function algorithm

        Arguments:
            None

        Returns:
            Q -- The Q function matrix
    """
    print("Q function")

    # Initialize variables
    results = np.zeros(iterations)
    Q = np.zeros((tests,k))
    N = np.zeros((tests,k))

    # Loop through the tests
    for j in range(tests):
        for i in range(iterations):
            if j >= len(Q):
                break

            # Choose the best club based on the Q function
            Qt = [Q[j, l] + math.sqrt(math.log(i + 1) / (N[j, l] + 1)) for l in range(k)]
            # Choose the best club based on the Q function
            A = np.argmax(Qt)

            # Calculate the distance and next state with random variation
            distance = golf_clubs[A] + randint(-1, 1)
            # Ensure the next state is within bounds
            next_state = min(j + distance, len(node_locs) - 1)
            # Get the reward for the next state
            reward = rewards[next_state]

            # add reward to results
            results[i] += reward
            # Increment the number of times the club has been used
            N[j, A] += 1
            # Update the Q function
            Q[j, A] = Q[j, A] + learning_rate * (reward + gamma * np.max(Q[next_state]) - Q[j, A])

    return Q

def UCB():
    """Function UCB implements the UCB algorithm

        Arguments:
            None

        Returns:
            Q -- The Q function matrix

    """
    print("UCB algorithm")
    # Initialize variables
    results = np.zeros(iterations)
    Q = np.zeros((tests, k))
    N = np.ones((tests, k))

    # Loop through the tests
    for j in range(tests):
        for i in range(iterations):
            # Compute UCB values
            Qt = [Q[j, l] + math.sqrt(math.log(i + 1) / (N[j, l] + 1)) for l in range(k)]
            # Select action with highest UCB value
            A = np.argmax(Qt)

            # Apply selected club with random swing variation
            distance = golf_clubs[A] + randint(-1, 1)
            # Ensure within bounds
            next_state = min(j + distance, len(node_locs) - 1)

            # Get the reward for landing on that marker
            reward = rewards[next_state]

            # Update the results
            results[i] += reward
            # Increment the number of times the club has been used
            N[j, A] += 1
            # Update the Q function
            Q[j, A] = Q[j, A] + (1 / N[j, A]) * (reward - Q[j, A])

    return Q

def run_plot():
    """Function run_plot to animate golf swings

        Arguments:
            None

        Returns:
            None
    """
    global quit
    global node_locs
    global inter
    global visited


    if quit or inter == (len(visited)-1):
        quit = True
        return
    create_branch(node_locs[visited[inter]][0],node_locs[visited[inter]][1],
                              node_locs[visited[inter+1]][0],node_locs[visited[inter+1]][1],myCanvas,"black","C"+str(inter))
    create_plot(myCanvas)


    if not quit:
        inter = inter + 1
        myCanvas.after(500, run_plot)

def close_properly(root):
    """Function close_properly to close the program properly

        Arguments:
            root -- The root of the program

        Returns:
            None
    """

    global quit
    if quit==False:
        quit = True
    else:
        root.destroy()

def on_closing():
    """Function on_closing to close the program properly

        Arguments:
            None

        Returns:
            None
    """
    global quit
    if quit==False:
        quit = True
    else:
        root.destroy()

root = Tk()
root.title("Golf - choose the correct club.")
map = ImageTk.PhotoImage(Image.open("golf_map.png"))
myCanvas = Canvas(root, width=width, height=height, borderwidth=0, highlightthickness=0, bg="white")
myCanvas.pack()
Canvas_Image = myCanvas.create_image(0,0, image=map, anchor="nw")

def main():
    """Function main to call and report the results

        Arguments:
            None

        Returns:
            None
    """
    global node_grah
    global visited
    create_plot(myCanvas)

    Button(root, text="Quit", command = lambda: close_properly(root)).pack()

    Q = UCB()
    #Q = Q_function()
    visited.append(0)
    #visited.append(0)
    reward = get_Q_path(Q)
    print("reward: " + str(reward))
    run_plot()
    print(Q)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()