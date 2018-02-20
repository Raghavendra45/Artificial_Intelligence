from copy import deepcopy
from time import time, sleep
import platform
import random
import StringIO
import sys
import os

if platform.system() != 'Windows':
    import resource

sys.path[0] = os.getcwd()


class Board:
    BLANK = 0
    NOT_MOVED = (-1, -1)
    __queen__ = None
    

    def __init__(self, player_1, player_2, width=7, height=7):
        self.width = width
        self.height = height

        self.queen = "queen"        

        self.__board_state__ = [[Board.BLANK for i in range(0, width)] for j in range(0, height)]
        self.__last_queen_move__ = Board.NOT_MOVED
        self.__queen_symbols__ = {Board.BLANK: Board.BLANK, self.queen: 'Q'}

        self.move_count = 0

        self.__player_1__ = player_1
        self.__player_2__ = player_2

        self.__active_player__ = player_1
        self.__inactive_player__ = player_2
	

    def get_state(self):
        return deepcopy(self.__board_state__)

    def __apply_move__(self, move):
        row,col = move
        self.__last_queen_move__ = move  
        self.__board_state__[row][col] = 'Q'

        #swap the players
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp       
      
        self.move_count = self.move_count + 1


    def __apply_move_write__(self, move):
        row, col = move
        self.__last_queen_move__ = move
        self.__board_state__[row][col] = 'Q'

        # swap the players
        tmp = self.__active_player__
        self.__active_player__ = self.__inactive_player__
        self.__inactive_player__ = tmp

        self.move_count = self.move_count + 1

    def copy(self):
        b = Board(self.__player_1__, self.__player_2__, width=self.width, height=self.height)
	b.__last_queen_move__ = self.__last_queen_move__            
        for key, value in self.__queen_symbols__.items():
            b.__queen_symbols__[key] = value
        b.move_count = self.move_count
        b.__active_player__ = self.__active_player__
        b.__inactive_player__ = self.__inactive_player__
	b.queen = self.queen 
        b.__board_state__ = self.get_state()
        return b

    def forecast_move(self, move):
        new_board = self.copy()
        new_board.__apply_move__(move)
        return new_board

    def get_active_player(self):
        return self.__active_player__

    def get_inactive_player(self):
        return self.__inactive_player__

 
    def get_legal_moves(self):
	return self.__get_moves__(self.__last_queen_move__)


    def __get_moves__(self, move):
        # Changed this function. Now the piece will move like QUEEN not like KING.
        
        
        if move == self.NOT_MOVED:
            return self.get_first_moves()
        

        r, c = move

        directions = [ (-1, -1), (-1, 0), (-1, 1),
                        (0, -1),          (0,  1),
                        (1, -1), (1,  0), (1,  1)]

        fringe = [((r+dr,c+dc), (dr,dc)) for dr, dc in directions 
                if self.move_is_legal(r+dr, c+dc)]

        valid_moves = []

        while fringe:
            move, delta = fringe.pop()
            
            r, c = move
            dr, dc = delta

            if self.move_is_legal(r,c):
                new_move = ((r+dr, c+dc), (dr,dc))
                fringe.append(new_move)
                valid_moves.append(move)

        return valid_moves



    def get_first_moves(self):
        return [(i, j) for i in range(0, self.height) for j in range(0, self.width) if
                self.__board_state__[i][j] == Board.BLANK]

    def move_is_legal(self, row, col):
        return 0 <= row < self.height and \
               0 <= col < self.width and \
               self.__board_state__[row][col] == Board.BLANK

    def get_player_locations(self):
        return [(i, j) for j in range(0, self.width) for i in range(0, self.height) if
                self.__board_state__[i][j] == 'Q']

    def print_board(self):

	p_r,p_c = self.__last_queen_move__
        b = self.__board_state__

        out = ''

        for i in range(0, len(b)):
            for j in range(0, len(b[i])):
                if not b[i][j]:
                    out += ' '

                elif i == p_r and j == p_c:
                    out += 'Q'
                
                else:
                    out += '-'

                out += ' | '
            out += '\n\r'

        return out

    def play_isolation_name_changed(self, time_limit=1000, print_moves=False):
        move_history = []

        if platform.system() == 'Windows':
            def curr_time_millis():
                return int(round(time() * 1000))
        else:
            def curr_time_millis():
                return 1000 * resource.getrusage(resource.RUSAGE_SELF).ru_utime

        while True:
            game_copy = self.copy()
            move_start = curr_time_millis()

            def time_left(): return time_limit - (curr_time_millis() - move_start)
            curr_move = Board.NOT_MOVED

            legal_player_moves = self.get_legal_moves()
            curr_move = self.__active_player__.move(game_copy, legal_player_moves,
                                                           time_left)             

            if curr_move is None:
                curr_move = Board.NOT_MOVED

            if self.__active_player__ == self.__player_1__:
		#print 'Appending move ',curr_move,' to player 1'
                move_history.append([curr_move])
            else:
		#print 'Appending move ',curr_move,' to player 2'
                move_history[-1].append(curr_move)

            if time_limit and time_left() <= 0:
                if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)
                return self.__inactive_player__, move_history, "timeout"

            legal_moves_of_queen = self.get_legal_moves()

            if curr_move not in legal_moves_of_queen:
                if print_moves:
                    print 'Winner: ' + str(self.__inactive_player__)
		return self.__inactive_player__, move_history, "illegal move"

            last_attempt = curr_move
	    self.__apply_move__(curr_move)
            


def game_as_text(winner, move_history, termination="", board=Board(1,2)):
    print(winner)
    #ans = io.StringIO()
    ans = StringIO.StringIO()
    k=0
   
    for i, move1 in enumerate(move_history):
        p1_move = move1[0]
        ans.write("player1 "+"%d." % i + " (%d,%d)\r\n" % p1_move)
        if p1_move != Board.NOT_MOVED:
            board.__apply_move_write__(p1_move)
        ans.write(board.print_board())

        if len(move1) > 1:
            p2_move = move1[1]
            ans.write("player2 "+"%d. ..." % i + " (%d,%d)\r\n" % p2_move)
            if p2_move != Board.NOT_MOVED:
                board.__apply_move_write__(p2_move)
            ans.write(board.print_board())
        k=k+1
    ans.write(termination + "\r\n")
    ans.write("Winner: " + str(winner) + "\r\n")

    return ans.getvalue()



def main():
    print("Starting game:")

    from grade_players import RandomPlayer
    from grade_players import TestAI

    board = Board(RandomPlayer(), TestAI())
    board_copy = board.copy()
    winner, move_history, termination = board.play_isolation_name_changed(time_limit=30000, print_moves=True)
    print game_as_text(winner, move_history, termination, board_copy)

if __name__ == '__main__':
    main()
