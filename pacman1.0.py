# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:33:59 2024

@author: clear
"""

import pygame
import sys
import numpy as np
import pickle
Pacman_animation = []   # PacMan animation images
ghosts_animation = []   # ghosts images
ghosts_animation.append(pygame.transform.scale(pygame.image.load('images/ghost_images/blue.png'), (20, 20)))
ghosts_animation.append(pygame.transform.scale(pygame.image.load('images/ghost_images/orange.png'), (20, 20)))
ghosts_animation.append(pygame.transform.scale(pygame.image.load('images/ghost_images/pink.png'), (20, 20)))
ghosts_animation.append(pygame.transform.scale(pygame.image.load('images/ghost_images/red.png'), (20, 20)))
for i in range(1, 5):
    Pacman_animation.append(pygame.transform.scale(pygame.image.load(f"images/player_images/{i}.png"), (20, 20)))
class Layout:
    def __init__(self, window_width, window_height, grid_size):
        self.window_width = window_width
        self.window_height = window_height
        self.grid_size = grid_size
        self.grid_width = window_width // grid_size
        self.grid_height = window_height // grid_size
        self.window_size = (window_width, window_height)
        self.game_grid = [[]]
        self.game_grid_old =[[]]
        # Define colors
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        # Generate the game grid
        self.generate_grid()

    def generate_grid(self):
        # 1 =  walls , 0 = points , 2 = pacman , 3 = blueky , 4 orangy , 5 = pinky , 6 = redy , 7 = fobbiden(inner walls) , empyty = -1
        self.game_grid =[
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
                        [1, 3, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        self.game_grid_old =[list(row) for row in self.game_grid]   # make a hard copy to keep track of eaten point
    def draw(self, window):
        for y, row in enumerate(self.game_grid):
            for x, tile in enumerate(row):
                rect = pygame.Rect(x * self.grid_size, y * self.grid_size, self.grid_size, self.grid_size)
                if tile == 1:   # walls    
                    pygame.draw.rect(window, self.BLUE, rect)
                elif self.game_grid_old[y][x] == 0:  # Point
                    self.game_grid[y][x] = 0
                    pygame.draw.circle(window, self.WHITE, (x * self.grid_size+self.grid_size//3, y *self.grid_size+self.grid_size//3), self.grid_size//3)
                else:
                    self.game_grid[y][x] = -1
    def is_wall(self, x, y):
        # Check if the specified position (x, y) is a wall in the maze
        row = y 
        col = x
        return self.game_grid[row][col] == 1 
        
    def set(self,position,type):
        self.game_grid[position.row][position.col] = type      # set ghost and pacman position ( removes points )
        
    def point_prev_state(self,position,type):
        prev_position = self.prev_state[type][0]
        
class Pacman:
    def __init__(self,start_Position,grid_size):
        self.position =start_Position
        self.direction = "right"  # Initial direction
        self.speed = 1  # Pacman's movement speed
        self.grid_size = grid_size
        self.type = 2
        self.score = 0
    def move(self):
        # Update Pacman's position based on the current direction
        if self.direction == "up":
            self.position.row -= self.speed
        elif self.direction == "down":
            self.position.row += self.speed
        elif self.direction == "left":
            self.position.col -= self.speed
        elif self.direction == "right":
            self.position.col += self.speed

    def change_direction(self, new_direction):
        # Change Pacman's direction
        self.direction = new_direction

    def eat_pellet(self):
        # Logic for Pacman eating a pellet
        pass

    def eat_power_pellet(self):
        # Logic for Pacman eating a power pellet
        pass

    def draw(self,image_index,window):
        # Draw Pacman on the screen

        # Update Pacman's angle based on the current direction
        angle = 0   # right
        if self.direction == "up":
            angle = 90
        elif self.direction == "down":
            angle = 270
        elif self.direction == "left":
            angle = 180
        curr_image = pygame.transform.rotate(Pacman_animation[image_index], angle)  # transform pacman              
        window.blit(curr_image, (self.position.get_col() * self.grid_size, self.position.get_row() * self.grid_size))
class Ghost:
    def __init__(self, start_position, type,grid_size):
        self.position = start_position   # current position
        self.type = type  # Ghost's type 3 = blue , 4 = orange , 5 = pink , 6 = red
        self.direction = "up"  # Initial direction
        self.speed = 1  # Ghost's movement speed
        self.grid_size = grid_size
        self.mode = [ 0,1,3,4]   # 0:normal , 2:scatter , 3:chassing , 4:dead
        self.timer = 0
        self.target_cell = Position(0,0)
        self.ghosts_position = []   # other ghots position
        self.locations =[Position(28,2),Position(1,6),Position(8,26),Position(25,26)]     
    def set_target_cell(self,target_cell):
        self.target_cell = target_cell

    def set_ghosts_pos(self,position1,position2,position3):
        self.ghosts_pos = [position1,position2,position3]

    def move(self):
        # Update Ghost's position based on the current direction
        if self.direction == "up":
            self.position.row -= self.speed
        elif self.direction == "down":
            self.position.row += self.speed
        elif self.direction == "left":
            self.position.col -= self.speed
        elif self.direction == "right":
            self.position.col += self.speed

    def change_direction(self, new_direction):
        # Change Ghost's direction
        self.direction = new_direction

    def draw(self,window):
        # Draw Pacman on the screen
        index = 0 # ( type 3)
        if self.type == 4:
            index = 1
        elif self.type == 5:
            index = 2
        elif self.type == 6:
            index = 3
        window.blit(ghosts_animation[index], (self.position.get_col() * self.grid_size, self.position.get_row() * self.grid_size))


    
    def is_valid_move(self,move , maze):  # returns true if move is succesful
        # "move" is  selected move ( 0 = left ,1 = right ,2 = up ,3 = down)
        
        ghost_speed = self.speed
        row = self.position.get_row()
        col = self.position.get_col()
        if move == "left":
            col -= ghost_speed
            return not(maze.is_wall(col, row))
        if move == "right":
            col += ghost_speed
            return not(maze.is_wall(col, row))
        if move == "up":
            row -= ghost_speed
            return not(maze.is_wall(col, row))
        if move == "down":
            row += ghost_speed
            return not(maze.is_wall(col, row))
    def reset_position(self):
        # Reset Ghost's position to starting position
        pass

    def normal(self):
        # Logic for Ghost to normal behaviour
        pass

    # move towards target cell
    def move_towards(self,maze):
        # Logic for Ghost to chasing Pacman
        # Calculate the direction to move towards Pac-Man
        dx = self.target_cell.get_col() - self.position.get_col()  # col distance
        dy = self.target_cell.get_row() - self.position.get_row()  # row distance
        temp_direction = ""

        # pacman found
        if dx == 0 and dy == 0: return
            
        # Choose the primary(shortest) axis to pursue Pac-Man
        if abs(dx) > abs(dy):     # prioritize moving horizontal
           
            if dx > 0 and self.is_valid_move("right" , maze):
                temp_direction = "right" 
            elif dx < 0 and self.is_valid_move("left" , maze):
                temp_direction = "left"
            elif dy > 0 and self.is_valid_move("down" , maze) :# select available vertical movements
                temp_direction = "down"
            elif self.is_valid_move("up" , maze):
                temp_direction = "up"
            
        else:                  # prioritize moving vertical
            
            if dy > 0 and self.is_valid_move("down" , maze):
                temp_direction = "down"     
            elif dy < 0 and self.is_valid_move("up" , maze):
                temp_direction = "up"
            elif dx > 0 and self.is_valid_move("right", maze):
                temp_direction = "right" 
            elif self.is_valid_move("left" , maze):
                temp_direction = "left"
        self.change_direction(temp_direction)
        self.move()
    def scatter(self):
        # Logic for Ghost exploring
        pass
    def chasing(self):
        # Logic for chasing pacman to running from Pacman
        pass
    def dead(self):
        # Logic for Ghost whenever 
        pass
    def is_found(self,maze):
        return maze[self.positio.row][self.positio.col] ==  2
class Position:
    def __init__(self, row, col):
        self.row = row  # row
        self.col = col  # column

    def distance_to(self, other_point):
        # Calculate the Euclidean distance between two points
        distance = ((self.row - other_point.row) ** 2 + (self.col - other_point.col) ** 2) ** 0.5
        return distance

    def get_row(self):
        return self.row
    def get_col(self):
        return self.col
def position_equal(position1,position2):
    return (position1.row ==position2.row) & (position1.col ==position2.col)
#Tile coding to extract features
class Tile:
    def __init__(self, start_cell, end_cell):
        self.start_cell = start_cell
        self.end_cell = end_cell

    def tile_state(self, game_grid):
        # Extract features from the maze within the specified region
        features = []
        s_row = self.start_cell.row
        s_col = self.start_cell.col
        e_row = self.end_cell.row
        e_col = self.end_cell.col
        for row in range(s_row, e_row ):
            for col in range(s_col, e_col):
                cell_value = game_grid[row][col]
                features.append(str(cell_value))  # Point
        # Encode features into a single state
        state = ''.join(features)
        return state
# partition blocks of th maze

#1st partition 
block_11 = Tile(Position(0,0),Position(7,7))
block_12 = Tile(Position(0,7),Position(7,14))
block_13 = Tile(Position(0,14),Position(7,21))
block_14 = Tile(Position(0,21),Position(7,28))
#2st partition 
block_21 = Tile(Position(7,0),Position(15,7))
block_22 = Tile(Position(7,7),Position(15,14))
block_23 = Tile(Position(7,14),Position(15,21))
block_24 = Tile(Position(7,21),Position(15,28))
#3st partition 
block_31 = Tile(Position(15,0),Position(23,7))
block_32 = Tile(Position(15,7),Position(23,14))
block_33 = Tile(Position(15,14),Position(23,21))
block_34 = Tile(Position(15,21),Position(23,28))
#4st partition 
block_41 = Tile(Position(23,0),Position(30,7))
block_42 = Tile(Position(23,7),Position(30,14))
block_43 = Tile(Position(23,14),Position(30,21))
block_44 = Tile(Position(23,21),Position(30,28))
# Define the blocks' positions
blocks = {
    "block_11": (Position(0, 0), Position(7, 7)),
    "block_12": (Position(0, 7), Position(7, 14)),
    "block_13": (Position(0, 14), Position(7, 21)),
    "block_14": (Position(0, 21), Position(7, 28)),
    "block_21": (Position(7, 0), Position(15, 7)),
    "block_22": (Position(7, 7), Position(15, 14)),
    "block_23": (Position(7, 14), Position(15, 21)),
    "block_24": (Position(7, 21), Position(15, 28)),
    "block_31": (Position(15, 0), Position(23, 7)),
    "block_32": (Position(15, 7), Position(23, 14)),
    "block_33": (Position(15, 14), Position(23, 21)),
    "block_34": (Position(15, 21), Position(23, 28)),
    "block_41": (Position(23, 0), Position(30, 7)),
    "block_42": (Position(23, 7), Position(30, 14)),
    "block_43": (Position(23, 14), Position(30, 21)),
    "block_44": (Position(23, 21), Position(30, 28))
}
# track pacman in insert into one of the black to get the current state of that block
def get_block(position):
    for block, (top_left, bottom_right) in blocks.items():
        if top_left.row <= position.row <= bottom_right.row and \
                top_left.col <= position.col <= bottom_right.col:
            return block
class Pallet():
    def __init__(self):
        self.reward = 10
        self.type = 0 
# check collision between ghost and pacman
def is_game_over(pacman, ghosts):
    game_over = False
    for i in ghosts:
        if position_equal(pacman,i):
            return True
    return game_over


class QLearningAgent:
    def __init__(self):
        self.states = {"00000000"}  # all possible states
        self.action = ["up","right","down","left"]
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        self.exploration_rate = 1
        self.decay_factor = 0.99887
        self.empty_space_penalty_factor = -0.1
        self.q_table = {}   #  'state': {'up': 'q-value, 'down': qv, 'right': qv,'left':qv}
        self.rewards = {
            0: 10,  # reward for point
            1: -50,  # penalty for walls
            3: -100,  # penalty for ghosts
            -1: -5,  # small penalty for moving into empty space
            2:0
        }

    def add_state(self,state):
        print(len(self.states),"\n")
        self.states.add(state)   # add the new state
        # initialize q-values of the new state
        self.q_table[state] = {"up": self.initialize_q_value(), "down": self.initialize_q_value(), "right": self.initialize_q_value(), "left": self.initialize_q_value()}

    def initialize_q_value(self):
        # Generate random Q-values for each action
        #q_value = np.random.rand(1)[0]
        return np.random.rand(1)[0]
    def get_reward(self,environment,pacman_position,last_point_position):
        distance = last_point_position.distance_to(pacman_position)   # distance from last point to pacman current position
        if environment == -1 :
            print("Empty  space ...","distance from last point :",distance,"\n","reward: ",self.empty_space_penalty_factor*distance,"\n")
            return self.empty_space_penalty_factor*distance
        print("reward: ",self.rewards[environment],"\n")
        return self.rewards[environment]
        
    def select_action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_rate:
            # Explore: Select a random action
            print("Explore \n")
            return np.random.choice(self.action)
        else:
            # Exploit: Select the action with the highest Q-value for the current state
            # Extract the dictionary associated with the target key
            target_dict = self.q_table[state]   # all possible action for the state
            action = max(target_dict, key=target_dict.get)   # action with highest q value
            print("Exploit :",target_dict," \n")
            return action
    def q_value(self,state, action):
        if state in self.q_table:
            value = self.q_table[state][action]
            return value

    def max_q_value(self,state):
        if state in self.q_table:
            target_dict = self.q_table[state]   # all possible action for the state
            action = max(target_dict, key=target_dict.get)   # action with highest q value
            return target_dict[action]

    def update_q_table(self, state, action, reward, next_state):
        # Q-learning update rule
        actions = self.q_table[state]
        old_q_value = self.q_value(state, action)
        max_next_q_value = self.max_q_value(next_state)
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - old_q_value)
        actions[action] = new_q_value
        self.q_table[state] = actions
        

# Example usage:
# Define the environment (grid world) with states and actions
# Initialize Q-learning agent
# agent = QLearningAgent(self)
# Train the agent by interacting with the environment
# for episode in range(num_episodes):
#     state = initial_state  # Start from initial state
#     while not is_terminal(state):  # Continue until reaching a terminal state
#         action = agent.select_action(state)  # Select action based on Q-values
#         next_state, reward = take_action(state, action)  # Execute action and observe next state and reward
#         agent.update_q_table(state, action, reward, next_state)  # Update Q-values
#         state = next_state  # Move to the next state



class GameWorld:
    def __init__(self):
        pygame.init()
        self.WINDOW_WIDTH = 560
        self.WINDOW_HEIGHT = 620
        self.GRID_SIZE = 20
        self.window = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.font = pygame.font.Font(None, 36)
        # Create a PacmanMaze instance
        self.maze = Layout(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.GRID_SIZE)
        self.pallet = Pallet()
        
        # number of episodes
        self.num_episodes = 0
        
        self.agent = QLearningAgent()   # for learning
        
        self.AIagent = PacManAgent()   # AI agent (trained)
        
        #initialize all chararacters
        self.pacman = Pacman(Position(28,2),self.maze.grid_size)
        self.inky = Ghost(Position(1,1),3,self.maze.grid_size)
        self.clyde = Ghost(Position(1,26),4,self.maze.grid_size)
        self.pinky = Ghost(Position(28,26),5,self.maze.grid_size)
        self.blinky= Ghost(Position(25,1),6,self.maze.grid_size)

        self.running = True
        self.pacman_animation_index = 0
        self.font = pygame.font.Font(None, 21)
        self.ghost_speed_controler = 0
        self.heart_img = pygame.transform.scale(pygame.image.load('images/heart.png'), (20, 20))

    def train(self):
        pygame.display.set_caption("Pacman")
        for i in range (self.num_episodes):
            print("Initializing   ... \n","Ipisode number :",i,"\n")
            # reset
            self.reset_game()
            self.running = True
            if i == self.num_episodes-1:
                save_q_table(self.agent.q_table)
            # set positions on the grid
            print("Episode number :",i,"\n","Decay factor :",self.agent.decay_factor,"\n")
            self.maze.set(self.pacman.position,self.pacman.type)
            self.maze.set(self.inky.position,3)
            self.maze.set(self.blinky.position,3)
            self.maze.set(self.pinky.position,3)
            self.maze.set(self.clyde.position,3)
            last_point_position = self.pacman.position
            while self.running:
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
            
            
                # check if state exist
                block = blocks[get_block(self.pacman.position)]   # get section pacman in in then get the topleft and bottom right positions
                tile = Tile(block[0], block[1])
                # learning
                state = tile.tile_state(self.maze.game_grid)
                # initialize reward
                reward = 0
                if state not in self.agent.states:
                    self.agent.add_state(state)
                
                # Update Pacman for the next iteration of animation
                self.pacman_animation_index = (self.pacman_animation_index + 1) % 4
                #maze.game_grid_old[pacman.position.row][pacman.position.col] = -1   # replace points with empty space
                # Handle player input for Pacman movement
                
                #take action
                action = self.agent.select_action(state)
                row = self.pacman.position.get_row() 
                col = self.pacman.position.get_col()
                
                # ghost's turn to make a move
             
                self.inky.set_target_cell(self.pacman.position) 
                self.blinky.set_target_cell(self.pacman.position)  
                self.pinky.set_target_cell(self.pacman.position) 
                self.clyde.set_target_cell(self.pacman.position)
                if self.ghost_speed_controler > 3 :  # pac man is 3 steps faster
                    self.inky.move_towards(self.maze)     
                    self.blinky.move_towards(self.maze)
                    self.pinky.move_towards(self.maze) 
                    self.clyde.move_towards(self.maze)   
                    self.ghost_speed_controler = 0 # reset 
                # ensure ghost did not run into pacman
                if is_game_over(self.pacman.position,[self.inky.position,self.blinky.position,self.pinky.position,self.clyde.position]):
                    self.running = False  
                    
                if action =="left":
                    col -= self.pacman.speed
                    reward = self.agent.get_reward(self.maze.game_grid[row][col],Position(row,col),last_point_position)
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("left")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            last_point_position = Position(row,col)
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point
                elif action =="right":
                    col += self.pacman.speed
                    reward = self.agent.get_reward(self.maze.game_grid[row][col],Position(row,col),last_point_position)
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("right")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            last_point_position = Position(row,col)
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point
                elif action =="up":
                    row -= self.pacman.speed
                    reward = self.agent.get_reward(self.maze.game_grid[row][col],Position(row,col),last_point_position)
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("up")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            last_point_position = Position(row,col)
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point
                elif action =="down":
                    row += self.pacman.speed
                    reward = self.agent.get_reward(self.maze.game_grid[row][col],Position(row,col),last_point_position)
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("down")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            last_point_position = Position(row,col)
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point 
                print("Pacman Position :",self.pacman.position.row," : ",self.pacman.position.col,"\nlast Point position : ",last_point_position.row," : ",last_point_position.col,"\n")

                self.window.fill(self.maze.BLACK)
            
               
                # Limit FPS
                pygame.time.Clock().tick(10)  # Adjust the argument for the desired frames per second
                self.maze.draw(self.window)
                
                self.maze.set(self.pacman.position,self.pacman.type)
                self.maze.set(self.inky.position,3)
                self.maze.set(self.blinky.position,3)
                self.maze.set(self.pinky.position,3)
                self.maze.set(self.clyde.position,3)
                
                # get section pacman in in then get the topleft and bottom right positions
                block = blocks[get_block(self.pacman.position)]  
                tile = Tile(block[0], block[1])
                # next state
                next_state = tile.tile_state(self.maze.game_grid)
                if next_state not in self.agent.states:
                    self.agent.add_state(next_state)
                #update q-table
                self.agent.update_q_table(state, action, reward, next_state)

                #  ensure pacman did not run into ghost
                # is_game_over(self.pacman.position,[self.inky.position,self.blinky.position,self.pinky.position,self.clyde.position]) or 
                if is_game_over(self.pacman.position,[self.inky.position,self.blinky.position,self.pinky.position,self.clyde.position]) or self.pacman.score ==2890 :
                    self.running = False
                # Draw the maze
                self.pacman.draw(self.pacman_animation_index,self.window)
                self.inky.draw(self.window)
                self.blinky.draw(self.window)
                self.pinky.draw(self.window)
                self.clyde.draw(self.window)
            
                # Draw score at the bottom
                display_text = self.font.render("  Episode Number : " + str(i)+"  |  Number of Sub-States Generated : " + str(len(self.agent.q_table))+"  |  Score: " + str(self.pacman.score), True, (255,255,255))
                display_rect = display_text.get_rect()
                display_rect.bottomleft = ( 0, self.WINDOW_HEIGHT)
                self.window.blit(display_text, display_rect)  

                # Update the display
                self.ghost_speed_controler +=1
                pygame.display.flip()
                
    def run(self):
        pygame.display.set_caption("Pacman")
        for i in range (10):
            # reset
            self.reset_game_run()
            self.running = True
            # set positions on the grid
            self.maze.set(self.pacman.position,self.pacman.type)
            self.maze.set(self.inky.position,3)
            self.maze.set(self.blinky.position,3)
            self.maze.set(self.pinky.position,3)
            self.maze.set(self.clyde.position,3)
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
            
            
                # check if state exist
                block = blocks[get_block(self.pacman.position)]   # get section pacman in in then get the topleft and bottom right positions
                tile = Tile(block[0], block[1])
                # learning
                state = tile.tile_state(self.maze.game_grid)
            
                
                # Update Pacman for the next iteration of animation
                self.pacman_animation_index = (self.pacman_animation_index + 1) % 4
                #maze.game_grid_old[pacman.position.row][pacman.position.col] = -1   # replace points with empty space
                # Handle player input for Pacman movement
                
                #take action
                action = self.AIagent.select_action(state)
                row = self.pacman.position.get_row() 
                col = self.pacman.position.get_col()
                
                # ghost's turn to make a move
             
                self.inky.set_target_cell(self.pacman.position) 
                self.blinky.set_target_cell(self.pacman.position)  
                self.pinky.set_target_cell(self.pacman.position) 
                self.clyde.set_target_cell(self.pacman.position)
                if self.ghost_speed_controler > 3 :  # pac man is 3 steps faster
                    self.inky.move_towards(self.maze)     
                    self.blinky.move_towards(self.maze)
                    self.pinky.move_towards(self.maze) 
                    self.clyde.move_towards(self.maze)   
                    self.ghost_speed_controler = 0 # reset speed
                    
                # ensure ghost did not run into pacman
                if is_game_over(self.pacman.position,[self.inky.position,self.blinky.position,self.pinky.position,self.clyde.position]):
                    self.running = False  
                
                if action =="left":
                    col -= self.pacman.speed
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("left")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point
                elif action =="right":
                    col += self.pacman.speed
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("right")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point
                elif action =="up":
                    row -= self.pacman.speed
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("up")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point
                elif action =="down":
                    row += self.pacman.speed
                    if not(self.maze.is_wall(col, row)):    # detect collision
                        self.pacman.change_direction("down")
                        self.pacman.move()
                        if self.maze.game_grid[self.pacman.position.row][self.pacman.position.col] == 0 :  # point detected
                            self.pacman.score+=self.pallet.reward
                            self.maze.game_grid_old[self.pacman.position.row][self.pacman.position.col] = -1  # remove point               
            
                self.window.fill(self.maze.BLACK)
                           
                # Limit FPS
                pygame.time.Clock().tick(10)  # Adjust the argument for the desired frames per second
                self.maze.draw(self.window)
                
                self.maze.set(self.pacman.position,self.pacman.type)
                self.maze.set(self.inky.position,3)
                self.maze.set(self.blinky.position,3)
                self.maze.set(self.pinky.position,3)
                self.maze.set(self.clyde.position,3)
                

                #  terminal state   reward < 0 for ghost and wall
                # is_game_over(self.pacman.position,[self.inky.position,self.blinky.position,self.pinky.position,self.clyde.position]) or 
                if is_game_over(self.pacman.position,[self.inky.position,self.blinky.position,self.pinky.position,self.clyde.position]) or self.pacman.score ==2890 :
                    self.running = False
                # Draw the maze
                self.pacman.draw(self.pacman_animation_index,self.window)
                self.inky.draw(self.window)
                self.blinky.draw(self.window)
                self.pinky.draw(self.window)
                self.clyde.draw(self.window)
                
               
            
                # Draw score at the bottom
                score_text = self.font.render(str(10-i)+"                                                  AIPacMman                 "+" Current Score: "  + str(self.pacman.score), True, (255,255,255))
                score_rect = score_text.get_rect()
                score_rect.bottomleft = ( 40, self.WINDOW_HEIGHT)
                self.window.blit(score_text, score_rect)
          
                #draw heart
                self.window.blit(self.heart_img, (5, 30*self.GRID_SIZE)) 

                # Update the display
                self.ghost_speed_controler +=1
                pygame.display.flip()
        # Quit Pygame
        pygame.quit()
        sys.exit()
    def reset_game_run(self):
        start_positions = [Position(28,2),Position(1,6),Position(8,26),Position(25,26)]        
        self.pacman.position = start_positions[round(np.random.rand()*3)]
           
    def reset_game(self):
        
        self.agent.exploration_rate = self.agent.exploration_rate*self.agent.decay_factor
        start_positions = [Position(28,2),Position(1,6),Position(8,26),Position(25,26)]
        self.maze = Layout(self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.GRID_SIZE)
        self.pallet = Pallet()
        
        self.pacman = Pacman(start_positions[round(np.random.rand()*3)],self.maze.grid_size)
        self.inky = Ghost(Position(1,1),3,self.maze.grid_size)
        self.clyde = Ghost(Position(1,26),4,self.maze.grid_size)
        self.pinky = Ghost(Position(28,26),5,self.maze.grid_size)
        self.blinky= Ghost(Position(25,1),6,self.maze.grid_size)
class PacManAgent:
    def __init__(self):
        self.q_table = {}
        self.action = ["up","right","down","left"]
   
        
    def select_action(self, state):
        # incase state was never seen
        if state not in self.q_table:
            return np.random.choice(self.action)
        else:
            # Exploit: Select the action with the highest Q-value for the current state
            # Extract the dictionary associated with the target key
            target_dict = self.q_table[state]   # all possible action for the state
            action = max(target_dict, key=target_dict.get)   # action with highest q value
            return action
        
def save_q_table(q_table):
    filename = "trained-pacman"
    with open(filename, 'wb') as file:
        pickle.dump(q_table, file)

def load_q_table():
    filename = "trained-pacman"
    with open(filename, 'rb') as file:
        q_table = pickle.load(file)
    return q_table

# Usage
# for learning usw QLearningAgent
# for playing use PacManAgent(Load "trained-pacman" first)
print(" Specify number of episodes to train the PacMan agent :")
num_episodes_input = int(input())
game = GameWorld()
game.num_episodes = num_episodes_input
game.train()
game.AIagent.q_table = load_q_table()   # load file
game.run()
