def visualize_nums(ifname='semiprimes30.txt'):
    with open(ifname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            (n, p, q, pq) = map(int, line.split())
            print(' '*15 + '{p:015b}   {p:10}'.format(p=p))
            print(' '*14 + '{q:016b}   {q:10}'.format(q=q))
            print('_'*43 + '*')
            print('{pq:30b}   {pq:10}'.format(pq=pq))
            print()

def main():
    visualize_nums()

if __name__ == '__main__':
    main()
