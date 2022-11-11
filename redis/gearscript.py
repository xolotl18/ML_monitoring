gb = GearsBuilder('KeysReader')
gb.map(lambda x : {'key' : x['key']. 'title' : execute('HGET', x['key'], 'title')})
gb.foreach(lambda x : execute('XADD', 'movieStream', 'code', x['key'], 'title', x['title']))
gb.register('movie:*')
