import random
import socket

class DnsRequest:
    def __init__(self, label, mx, ns):
        if mx and ns:
            raise Exception("mx and ns cannot both be specified")
        self.pid = random.getrandbits(16).to_bytes(2, 'big')
        self.header = self.pid + b'\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00'
        self.qname = b''.join(len(x).to_bytes(1, 'big') + bytes(x, 'utf-8') for x in label.split('.'))+b'\0'
        self.qtype = b'\x00\x0f' if mx else (b'\x00\x02' if ns else b'\x00\x01')
        self.qclass = b'\x00\x01'
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def request(self):
        return self.header + self.qname + self.qtype + self.qclass
    def send(self, host, port):
        self.socket.connect((host, port))
        msg = self.request()
        msgleft = len(msg)
        while msgleft > 0:
            sent = self.socket.send(msg[-msgleft:])
            if sent == 0:
                raise RuntimeError("socket connection broken")
            msgleft -= sent
    def recv(self):
        recvd = self.socket.recv(2048)
        print(recvd)
        

dns = DnsRequest("www.mcgill.ca", False, False)
print(dns.request())
dns.send('8.8.8.8', 53)
dns.recv()
