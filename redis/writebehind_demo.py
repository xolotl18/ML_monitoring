from rgsync import RGWriteBehind, RGWriteThrough
from rgsync.Connectors import MySqlConnector, MySqlConnection

'''
Create MySQL connection object
'''
connection = MySqlConnection(user='giacomo', passwd='Hello!world1996', db='host.docker.internal/movieDB')

'''
Create MySQL persons connector
'''
moviesConnector = MySqlConnector(connection=connection, tableName='movies', pk='movieId')

moviesMappings = {
	'title':'title',
	'year':'year',
}

RGWriteBehind(GB,  keysPrefix='movie', mappings=moviesMappings, connector=moviesConnector, name='MoviesWriteBehind',  version='99.99.99')

