import berserk
import datetime

def download_games(user_name, max_games=10, days_back=10):
    f = open("lichess.token", "r")
    token = (f.read())
    token = token.strip()
    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    td0 = datetime.date.today()
    td = td0 + datetime.timedelta(days=1)
    tdback = td - datetime.timedelta(days=days_back)

    tdy = int(td.strftime("%Y"))
    tdm = int(td.strftime("%m"))
    tdd = int(td.strftime("%d"))

    tdybk = int(tdback.strftime("%Y"))
    tdmbk = int(tdback.strftime("%m"))
    tddbk = int(tdback.strftime("%d"))

    start = berserk.utils.to_millis(datetime.datetime(tdybk, tdmbk, tddbk))
    end = berserk.utils.to_millis(datetime.datetime(tdy, tdm, tdd))

    gms = client.games.export_by_player(user_name, since=int(start), until=int(end), max=max_games, as_pgn=True, perf_type='bitz')
    games_list = [str(elem) for elem in list(gms)]
    print(f'Downloaded {len(games_list)} games')
    return ' '.join(games_list)
