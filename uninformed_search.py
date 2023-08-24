import math
import random
import copy
from collections import deque
from itertools import count

# N-Queens Problem


# calculate the total number of possible queen placements on n x n chessboard
def num_placements_all(n):
    # 8 -> 8x8 : 64 (8) * 63 (7) * 62 (6) * 61 (5) * 60 (4) * 59 (3) * 58 (2)
    # * 57 (1) -> (8^2)! / (8^2 - 8))!
    num = math.factorial(n ** 2)
    den = math.factorial(n ** 2 - n)
    return num / den

# calculate the total number of possible queen placements on n x n chessboard
# with 1 queen per row
def num_placements_one_per_row(n):
    return n ** n

# check if the given board configuration is a valid solution
def n_queens_valid(board):
    length = len(board)
    tracker = {}
    
    for i in range(0, length):
        # make sure column isn't the same
        if board[i] in tracker.values():
            return False
        # check diagonals, can find that if the difference between
        # rows and columns, respectively, are equal to each other
        for j in range(0, len(tracker)):
            if (abs(tracker[j] - board[i]) == abs(j - i)):
                return False
        # if no conflicts, add new entry to our tracker
        else:
            tracker[i] = board[i]
    return True

# find all solutions on n x n chessboard
def n_queens_solutions(n):
    solutions = []
    def solver(board):
        if len(board) == n:
            solutions.append(board[:])  # store a valid solution
            return

        for col in range(n):
            board.append(col)
            if n_queens_valid(board):
                solver(board)
            board.pop()  # clear the position for backtracking

    board = []
    solver(board)
    return solutions

# Lights Out

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board

    def get_board(self):
        return self.board

    def __setitem__(self, key1, key2, value):
        self.board[key1][key2] = value

    def perform_move(self, row, col):
        if row > 0:
            self.board[row - 1][col] = not self.board[row - 1][col]
        if row < len(self.board) - 1:
            self.board[row + 1][col] = not self.board[row + 1][col]
        if col > 0:
            self.board[row][col - 1] = not self.board[row][col - 1]
        if col < len(self.board[0]) - 1:
            self.board[row][col + 1] = not self.board[row][col + 1]
        self.board[row][col] = not self.board[row][col]

    def scramble(self):
        for x in range(0, len(self.board)):
            for y in range(0, len(self.board[0])):
                if (random.random() < 0.5):
                    self.perform_move(x, y) 
        

    def is_solved(self):
        for x in self.board:
            for y in x:
                if y == True:
                    return False
        return True

    def copy(self):
        # deep copy rather than shallow copying
        return copy.deepcopy(self)

    def successors(self):
        for x in range(len(self.board)):
            for y in range(len(self.board[0])):
                thing = self.copy()
                thing.perform_move(x, y)
                yield ((x, y), thing)

    def find_solution(self):
        visited = set()
        queue = deque()
        parent = {}
        moves = {}
        parent[self] = None
        moves[self] = None

        queue.append(self)
        visited.add(tuple(tuple(x) for x in self.get_board()))

        while queue:
            puzzle_instance = queue.popleft()

            if puzzle_instance.is_solved():
                node = puzzle_instance
                solution = []
                while parent[node]:
                    solution.append(moves[node])
                    node = parent[node]
                return list(reversed(solution))

            for move, neighbor in puzzle_instance.successors():
                neighbor_board = tuple(tuple(x) for x in neighbor.get_board())
                if neighbor_board not in visited:
                    parent[neighbor] = puzzle_instance
                    moves[neighbor] = move
                    if neighbor.is_solved():
                        node = neighbor
                        solution = []
                        while parent[node]:
                            solution.append(moves[node])
                            node = parent[node]
                        return list(reversed(solution))
                    queue.append(neighbor)
                    visited.add(neighbor_board)

        return None
        
                    
                

def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False for cols in range(0, cols)] \
                               for row in range(0, rows)])
                                
# Linear Disk Movement

class DiskMovement(object):
    def __init__(self, disks, length, n):
        # Initialize the DiskMovement object with a list of disks, length, and n
        self.disks = list(disks)
        self.length = length
        self.n = n

    def successors(self):
        # Generate all possible successor states by moving disks to adjacent empty slots
        for i, disk in enumerate(self.disks):
            if disk != 0:
                for j in range(-2, 3):
                    if j != 0 and 0 <= i + j < self.length and self.disks[i + j] == 0 and (j == 1 or self.disks[i + j - 1] != 0):
                        temp = list(self.disks)
                        temp[i], temp[i + j] = 0, disk
                        # Yield the move and the new DiskMovement object representing the successor state
                        yield ((i, i + j), DiskMovement(temp, self.length, self.n))

def is_solved_identical(dm):
    # Check if configuration is solved for identical disks (all  last n disks have the value of 1)
    return all(disk == 1 for disk in dm.disks[-dm.n:])

def is_solved_distinct(dm):
    # Check if the configuration is solved for distinct disks (last dm.n elements are in ascending order)
    return all(dm.disks[-i - 1] == i + 1 for i in range(dm.n))

def solve_identical_disks(length, n, initial_distinct_disk=[]):
    # Create a dm object based on the initial configuration of disks
    dm = DiskMovement(
        initial_distinct_disk if initial_distinct_disk else [1]*n + [0]*(length-n),
        length, n
    )
    moves, parent = {dm: ()}, {dm: dm}
    q, explored_set = deque([dm]), set([tuple(dm.disks)])

    def get_solution(node):
        # Retrieve the sequence of moves from the initial state to given node
        solution = []
        while parent[node] != node:
            solution.append(moves[node])
            node = parent[node]
        return list(reversed(solution))

    while q:
        diskInstance = q.popleft()
        if is_solved_identical(diskInstance) or is_solved_distinct(diskInstance):
            # Return the sequence of moves to the goal state
            return get_solution(diskInstance)
        for move, neighbor in diskInstance.successors():
            if tuple(neighbor.disks) not in explored_set:
                parent[neighbor] = diskInstance
                moves[neighbor] = move
                explored_set.add(tuple(neighbor.disks))
                q.append(neighbor)
    return None

def solve_distinct_disks(length, n):
    # Create an initial configuration for distinct disks and call the solve_identical_disks function.
    initialDisks = [i+1 for i in range(n)] + [0] * (length - n)
    return solve_identical_disks(length, n, initialDisks)

