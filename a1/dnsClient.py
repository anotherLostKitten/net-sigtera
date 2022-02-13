import random
import socket
import sys
import getopt
import re
import time

class DnsRequest:
    def __init__(self, label, qtype):
        self.pid = random.getrandbits(16).to_bytes(2, 'big')
        self.flags = b'\x01\x00'
        self.qdcount = b'\x00\x01'
        self.ancount = b'\x00\x00'
        self.nscount = b'\x00\x00'
        self.arcount = b'\x00\x00'
        self.qname = b''.join(len(x).to_bytes(1, 'big') + bytes(x, 'utf-8') for x in label.split('.'))+b'\0'
        self.qtype = {'A': b'\x00\x01', 'NS': b'\x00\x02', 'MX': b'\x00\x0f'}[qtype]
        self.qclass = b'\x00\x01'

    def request(self):
        return self.pid + self.flags + self.qdcount + self.ancount + self.nscount + self.arcount + self.qname + self.qtype + self.qclass

class DnsResponse:
    def __init__(self, msg):
        self.pid = msg[0:2]
        self.flags = msg[2:4]
        self.qdcount = int.from_bytes(msg[4:6], 'big')
        self.ancount = int.from_bytes(msg[6:8], 'big')
        self.nscount = int.from_bytes(msg[8:10], 'big')
        self.arcount = int.from_bytes(msg[10:12], 'big')
        self.qds = []
        self.ans = []
        self.ars = []
        i = 12
        for j in range(0, self.qdcount):
            (lbl, di) = self.decode_labels(msg, i)
            i += di
            self.qds.append((lbl, msg[i:i+2], msg[i+2:i+4]))
            i += 4
        for j in range(0, self.ancount):
            (vals, di) = self.decode_answer(msg, i)
            self.ans.append(vals)
            i += di
        for j in range(0, self.nscount):
            i = self.skip_answer(msg, i)
        for j in range(0, self.arcount):
            (vals, di) = self.decode_answer(msg, i)
            self.ars.append(vals)
            i += di

        self.check_flags()
            
    def decode_labels(self, msg, i):
        if msg[i] == 0:
            return (b'\0', 1)
        if msg[i] >= 192:
            ptr = int.from_bytes(msg[i:i+2], 'big') & 16383
            return (self.decode_labels(msg, ptr)[0], 2)
        (ret, di) = self.decode_labels(msg, i + msg[i] + 1)
        return (msg[i:i + msg[i] + 1] + ret, di + msg[i] + 1)

    def decode_ip(self, msg, i, ir):
        return f"{msg[i]}.{msg[i+1]}.{msg[i+2]}.{msg[i+3]}"

    def decode_ns(self, msg, i, ir):
        return self.stringify_labels(self.decode_labels(msg, i)[0])

    def decode_mx(self, msg, i, ir):
        pref = int.from_bytes(msg[i:i+2], 'big')
        exch = self.decode_ns(msg, i+2, ir)
        return f"{exch}\t{pref}"

    def stringify_labels(self, lbl):
        i = 0
        txt = ""
        while lbl[i] != 0:
            txt += lbl[i+1:i+lbl[i]+1].decode('ascii') + "."
            i+=lbl[i]+1
        return txt[:-1]

    
    def decode_answer(self, msg, i):
        (lbl, di) = self.decode_labels(msg, i)
        i += di
        rdlength = int.from_bytes(msg[i+8:i+10], 'big')
        (rdtype, rdfn) = (None, None)
        try:
            (rdtype, rdfn) = {
                b'\0\x01': ("IP", self.decode_ip),
                b'\0\x02': ("NS", self.decode_ns),
                b'\0\x05': ("CNAME", self.decode_ns),
                b'\0\x0f': ("MX", self.decode_mx)
            }[msg[i:i+2]]
        except:
            raise Exception(f"ERROR\tUnrecognized response type {msg[i:i+2].hex()}")
        if msg[i+2:i+4] != b'\0\x01':
            raise Exception(f"ERROR\tUnexpected response class {msg[i+2:i+4]}, shoud be 0001")
        return ((self.stringify_labels(lbl), rdtype, msg[i+2:i+4], int.from_bytes(msg[i+4:i+8], 'big'), rdlength, rdfn(msg, i+10, i+10+rdlength)), di+10+rdlength)

    def skip_answer(self, msg, i):
        i += self.decode_labels(msg, i)[1]
        rdlength = int.from_bytes(msg[i+8:i+10], 'big')
        return i+10+rdlength
    def get_auth(self):
        return "auth" if self.flags[0] & 4 != 0 else "noauth"

    def check_flags(self):
        if self.flags[0]&128 == 0:
            raise Exception("ERROR\tUnexpected response was a query")
        if self.flags[1]&128 == 0:
            print("ERROR\tServer does not allow recursive queries")
        if self.flags[1]&15 == 1:
            raise Exception("ERROR\tFormat error in sent packet")
        elif self.flags[1]&15 == 2:
            raise Exception("ERROR\tServer failure")
        elif self.flags[1]&15 == 3:
            raise Exception("NOT FOUND")
        elif self.flags[1]&15 == 4:
            raise Exception("ERROR\tQuery type not implemented by server")
        elif self.flags[1]&15 == 5:
            raise Exception("ERROR\tServer refused query")
        
    
    def print(self):
        
        auth = self.get_auth()
        if self.ans:
            print(f"***Answer Section ({self.ancount} records)***")
            for (q, rdtype, rdclass, ttl, rdlen, rddata) in self.ans:
                print(f"{rdtype}\t{rddata}\t{ttl}\t{auth}")
        else:
            print("NOT FOUND")
        if self.ars:
            print(f"***Answer Section ({self.arcount} records)***")
            for (q, rdtype, rdclass, ttl, rdlen, rddata) in self.ars:
                print(f"{rdtype}\t{rddata}\t{ttl}\t{auth}")
            

def send(s, dns):
    msg = dns.request()
    msgleft = len(msg)
    while msgleft > 0:
        sent = s.send(msg[-msgleft:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        msgleft -= sent

def recv(s, t):
    s.settimeout(t)
    return s.recv(2048)

def main(host, query, vals):
    print(f"DNS Client sending request for\n{query} Server: {host}\nRequest type: {vals['type']}")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((host, vals['p']))
        retries = 0
        reqs = {}
        resp = b''
        start_time = time.time()
        for i in range(0, vals['r']):
            dns = DnsRequest(query, vals['type'])
            reqs[dns.pid] = dns
            send(s, dns)
            try:
                resp = recv(s, vals['t'])
                break
            except socket.timeout:
                retries += 1
        if resp == b'':
            raise Exception(f"ERROR\tMaximum number of retries {retries} exceeded")
        elapsed = time.time() - start_time
        print(f"Response recieved after {elapsed} seconds ({retries} retries)")
        dns = DnsResponse(resp)
        if dns.pid not in reqs.keys():
            raise Exception("ERROR\tUnexpected response ID does not match any sent packets")
        dns.print()

if __name__ == "__main__":
    try:
        (opts, args) = getopt.getopt(sys.argv[1:], 't:r:p:n:m:')
        assert len(args) == 2, f"should have 2 arguments, has {len(args)}"
        assert re.fullmatch('@[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', args[0]), "specify DNS server with the form @x.x.x.x"
        host = args[0][1:]
        query = args[1]
        vals = {'p': 53,
                't': 5,
                'r': 3,
                'type': 'A'}
        for (o, v) in opts:
            if o in ['-p', '-r', '-t']:
                assert v.isdigit(), f"Argument for {o} must be an integer"
                vals[o[1]] = int(v)
            elif o == '-m':
                assert v == 'x', f"option -m{v} not recognized"
                assert vals['type'] == 'A', f"cannot specify both -ns and -mx"
                vals['type'] = 'MX'
            elif o == '-n':
                assert v == 's', f"option -n{v} not recognized"
                assert vals['type'] == 'A', f"cannot specify both -ns and -mx"
                vals['type'] = 'NS'    
        main(host, query, vals)
    except getopt.GetoptError as e:
        print(f"ERROR\tIncorrect input syntax: {e}")
    except AssertionError as e:
        print(f"ERROR\tIncorrect input syntax: {e}")
    except Exception as e:
        print(e)
