# This CommandReader will return the number of arguments provided to it
gb = GB('CommandReader')
gb.map(lambda x: {
    'key': x[0],
    'arg1': x[1],
    'arg2': x[2]
    })
gb.foreach(lambda x: execute('AI.TENSORSET', 'test-tensor', 'FLOAT', '2', '2','VALUES', '1', '2', '3', '4'))
gb.register(trigger='getfft')
