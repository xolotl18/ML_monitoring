import ml2rt
import redisai as rai

con = rai.Client(host='localhost', port=6379)

loaded_s = ml2rt.load_model('dummy.pt')
con.modelstore('model', 'torch', 'cpu', data=loaded_s)
