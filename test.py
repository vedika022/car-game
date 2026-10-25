import math

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("---------")
    print()


def check_winner(board):
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != ' ':
            return board[i][0]

        if board[0][i] == board[1][i] == board[2][i] != ' ':
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] != ' ':
        return board[0][2]

    return None


def is_full(board):
    return all(cell != ' ' for row in board for cell in row)


def minimax(board, maximizingPlayer):
    winner = check_winner(board)

    # Terminal states
    if winner == 'X':
        return 1
    elif winner == 'O':
        return -1
    elif is_full(board):
        return 0

    # Maximizing player (AI: X)
    if maximizingPlayer:
        best_score = -math.inf

        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    print(f"Testing AI move at ({i}, {j})")         

                    score = minimax(board, False)

                    board[i][j] = ' '
                    best_score = max(best_score, score)

        return best_score

    # Minimizing player (Human: O)
    else:
        best_score = math.inf

        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    print(f"Testing Player move at ({i}, {j})")

                    score = minimax(board, True)

                    board[i][j] = ' '
                    best_score = min(best_score, score)

        return best_score


def best_move(board):
    best_score = -math.inf
    move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'

                print(f"Evaluating move at ({i}, {j})")
                score = minimax(board, False)

                board[i][j] = ' '

                if score > best_score:
                    best_score = score
                    move = (i, j)

    print(f"AI chooses move at {move} with score {best_score}")
    return move


def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]

    print("Initial board:")
    print_board(board)

    while True:
        # Player's move
        try:
            player_move = tuple(
                map(int, input("Enter your move (row and column): ").split())
            )

            if len(player_move) != 2:
                raise ValueError

            row, col = player_move

            if row not in range(3) or col not in range(3):
                raise ValueError

            if board[row][col] == ' ':
                board[row][col] = 'O'
            else:
                print("Invalid move! Try again.")
                continue

        except (ValueError, IndexError):
            print("Invalid input! Enter row and column between 0 and 2.")
            continue

        print("Board after player's move:")
        print_board(board)

        if check_winner(board) or is_full(board):
            break

        # AI's move
        ai_move = best_move(board)

        if ai_move:
            board[ai_move[0]][ai_move[1]] = 'X'

        print("Board after AI's move:")
        print_board(board)

        if check_winner(board) or is_full(board):
            break

    winner = check_winner(board)

    if winner:
        print(f"Winner: {winner}")
    else:
        print("It's a tie!")


# Play the game
play_game()