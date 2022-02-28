"""
Collection of the core mathematical operators used throughout the code base.
"""


import math
# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x * y


def id(x):
    ":math:`f(x) = x`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x


def add(x, y):
    ":math:`f(x, y) = x + y`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x + y


def neg(x):
    ":math:`f(x) = -x`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return x * -1.0


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if x > y:
        return x
    else:
        return y


def is_close(x, y):
    ":math:`f(x) = |x - y| < 1e-2` "
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if abs(x - y) < .001:
        return True
    else:
        return False


def sigmoid(x: float):
    """
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if x >= 0:
        return (1.0 / (1.0 + math.exp(-1 * x)))
    else:
        return (math.exp(x) / (1.0 + math.exp(x)))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if x > 0.0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return d / (x + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return 1.0 / x


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    return -(1.0 / x ** 2) * d


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    # raise NotImplementedError('Need to implement for Task 0.1')
    if x > 0:
        return d
    else:
        return 0

# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    # def loop(ls):
    #     newList = []
    #     for i in range(len(ls)):
    #         newList.append(fn(ls[i]))
    #     return newList
    # return loop
    return lambda list : [fn(element) for element in list]


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    # def twolists(ls1, ls2):
    #     ls3 = []
    #     for i in range(len(ls1)):
    #         ls3.append(fn(ls1[i], ls2[i]))
    #     return ls3
    # return twolists
    return lambda ls1, ls2 : [fn(ls1_element, ls2_element) for ls1_element, ls2_element in zip(ls1, ls2)]


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    """
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 divides by ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    def red(ls):
        if(len(ls) > 1):
            beg = fn(ls[start + 1], ls[start])

            for x in range(start + 2, len(ls)):
                beg = fn(ls[x], beg)
            return beg
        elif len(ls) == 1:
            return fn(0, ls[0])
        else:
            return 0
    return red


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return reduce(add, 0)(ls)


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    # TODO: Implement for Task 0.3.
    # raise NotImplementedError('Need to implement for Task 0.3')
    return reduce(mul, 0)(ls)
