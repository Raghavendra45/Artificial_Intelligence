from random import randint


class RandomPlayer():
    """Player that chooses a move randomly."""    

    def move(self, game, legal_moves, time_left):
        if not legal_moves: return (-1,-1)            
        moves = legal_moves[randint(0,len(legal_moves)-1)]
        return moves
    


class HumanPlayer():
    """Player that chooses a move according to
    user's input."""
    def move(self, game, legal_moves, time_left):
        i=0
        choice = {}
        if not len(legal_moves):
            return None
       
        for move in legal_moves:
	    choice.update({i:move})
	    if (i + 1) % 6 == 0:
	        print '\t'.join(['[%d] : (%d,%d)' % (i,move[0],move[1])])
	    else:
	        print '\t'.join(['[%d] : (%d,%d)' % (i, move[0], move[1])]),
	    i += 1
	print
        print game.print_board()
        valid_choice = False
        while not valid_choice:
            try:
                index = int(input('Select move index:'))
                valid_choice = 0 <= index < i

                if not valid_choice:
                    print('Illegal move! Try again.')
            
            except ValueError:
                print('Invalid index! Try again.')
        game.__board_state__
        return choice[index]
