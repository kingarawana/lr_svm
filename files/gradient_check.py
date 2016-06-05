import numpy as np
from random import randrange

def grad_check(f, w, analytic_grad, num_checks):
  h = 1e-5
  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in w.shape])

    w[ix] += h
    fwph = f(w)
    w[ix] -= 2 * h
    fwmh = f(w)
    w[ix] += h
    grad_numerical = (fwph - fwmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    print 'numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error)

