from IPy import IP

def truncate(ip, prefix):
    if '/' in ip:
        ip = ip.split('/')[0]
    return IP(ip+'/'+str(prefix), make_net=True).strNormal()

def toBin(ip):
    ip = IP(ip, make_net=True)
    return ip.int()

def toStr(ip, prefix):
    ip = IP(ip).strNormal()
    return IP(ip+'/'+str(prefix), make_net=True).strNormal()

