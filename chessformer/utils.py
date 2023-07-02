def is_promotion(pretokens):
    for elt in pretokens:
        if elt.startswith('f_'):
            move_from = elt[2:]
            break
    if f'p_{move_from}' in pretokens or f'P_{move_from}' in pretokens:
        for elt in pretokens:
            if elt.startswith('t_'):
                move_to = elt[2:]
                break
        if move_to[1] == '1' or move_to[1] == '8':
            return True
    return False