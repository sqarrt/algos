import random


def erato_sieve(n):
    prime = [True, ] * (n+1)
    prime[0] = prime[1] = False
    for i in range(2, n):
        if prime[i] and i*i <= n:
            for j in range(i*i, n, i):
                prime[j] = False
    return [i for i, v in enumerate(prime[:n]) if v]


def _linear_erato_sieve(n):
    lp = [0, ] * (n + 1)
    pr = []

    for i in range(2, n):
        if not lp[i]:
            lp[i] = i
            pr.append(i)
        j = 0
        while j < len(pr) and pr[j] <= lp[i]:
            lp_ = i * pr[j]
            if lp_ < n:
                lp[lp_] = pr[j]
            else:
                break
            j += 1
    return pr, lp


def linear_erato_sieve(n):
    return _linear_erato_sieve(n)[0]


def factorization(n):
    result = []
    sieve = _linear_erato_sieve(n+1)[1]

    curNum = n
    while curNum != 1:
        result.append(sieve[curNum])
        curNum = int(curNum / sieve[curNum])

    return result


def binpow(a, n):
    res = None
    while n:
        if n % 2 == 1:
            res = res * a if res else a
        a *= a
        n = n >> 1
    return res or 1


def euler(n):
    res = n

    for i in factorization(n):
        if n % i == 0:
            while n % i == 0 and n != i:
                n /= i
            res -= res/i

    return int(res)


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


# некорректно
def gcd_ext(a, b):
    if a == 0:
        x = 0
        y = 1
        return b, x, y
    d, x1, y1 = gcd_ext(b % a, a)
    x = y1 - (b / a) * x1
    y = x1
    return d, x, y


class _matrix():
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def height(self):
        return len(self.matrix)

    @property
    def width(self):
        return len(self.matrix[0])

    def __mul__(self, other):
        if self.width != other.height:
            raise Exception('Cannot multiply those matrixes')

        res = _matrix([[None, ]*self.height for _ in range(self.width)])
        for i in range(len(res)):
            for j in range(len(res[i])):
                res[i, j] = sum(self[i,r]*other[r,j] for r in range(self.width))

        return res

    def __getitem__(self, item):
        if isinstance(item, tuple):
            i, j = item
            return self.matrix[i][j]
        return self.matrix[item]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            i, j = key
            self.matrix[i][j] = value
            return
        self.matrix[key] = value

    def __len__(self):
        return len(self.matrix)


def fib(n):
    matrix = _matrix([(0, 1),
                      (1, 1)])
    return binpow(matrix, n + 1)[0, 0]


def fib_irr(n):
    sq5 = 5**(1/2)
    return int((((1+sq5)/2)**n - ((1-sq5)/2)**n)/sq5)


def gray(n):
    return n ^ (n >> 1)


class Tree:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __add__(self, other):
        if not isinstance(other, Tree):
            other = Tree(other)
        return self.__add_tree__(other)

    def __add_tree__(self, other):
        t1 = self
        t2 = other
        if t1.value < t2.value:
            t1, t2 = t2, t1
        if t1.left:
            t1.left, t1.right = t1.right, t1.left
        if not t1.left:
            t1.left = t2
        else:
            t1.left = t1.left.__add_tree__(t2)
        return t1

    def str_recursive(self, offset):
        res = str(self.value)
        if self.left:
            left = f'-L->{self.left.str_recursive(offset+4)}'
            res += '\n' + ' '*offset + left
        if self.right:
            right = f'-R->{self.right.str_recursive(offset+4)}'
            res += '\n' + ' '*offset + right
        return res

    def __str__(self):
        return self.str_recursive(2)


def merge_sort(list_):
    l = len(list_)
    if l == 1:
        return list_
    if l == 2:
        if list_[0] > list_[1]:
            list_ = [list_[1], list_[0]]
        return list_
    equator = l // 2
    left = merge_sort(list_[:equator])
    right = merge_sort(list_[equator:])
    res = []
    for i in range(l):
        if not left:
            left = right
        if left[-1] < right[-1]:
            left, right = right, left

        res.append(left.pop())

    return list(reversed(res))


def quick_sort(list_):
    l = len(list_)
    if l < 2:
        return list_
    base = random.choice(list_)

    le = [el for el in list_ if el < base]
    ge = [el for el in list_ if el >= base]

    return quick_sort(le)+quick_sort(ge)


def intToRoman(num: int) -> str:
        symbol_values = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }

        good_replacement = {
            'IIII': 'IV',
            'XXXX': 'XL',
            'CCCC': 'CD',
            'VIIII': 'IX',
            'LXXXX': 'XC',
            'DCCCC': 'CM',

        }

        symbol_cnts = {}

        for sym, val in reversed(symbol_values.items()):
            num, cnt = num % val, num // val
            symbol_cnts[sym] = cnt

        res = []

        for sym, cnt in symbol_cnts.items():
            to_add = sym * cnt
            last = res.pop() if res else ''
            to_add = good_replacement.get(last + to_add,
                     last + good_replacement.get(to_add, to_add))
            res += to_add

        return ''.join(res)
