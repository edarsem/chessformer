import berserk
import datetime

def download_games(user_name, days_back=2):
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
    user = user_name

    gms = client.games.export_by_player(user, since=int(start), until=int(end), max=500, as_pgn=True)
    listToStr = ' '.join([str(elem) for elem in list(gms)])
    return listToStr