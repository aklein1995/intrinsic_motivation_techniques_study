import numpy as np

def yvalue_richard_curve(im_coef,im_type,max_steps,timestep):
    """
        https://en.wikipedia.org/wiki/Generalised_logistic_function
    """
    # print(max_steps)
    # print(timestep)
    K = im_coef
    A = K/100
    B = 0.5
    v = 0.05
    Q = 1
    C = 1
    # ratio = 16/max_steps
    # t = 16 - ((timestep+1)*ratio)
    # y = A + (K-A)/((C+Q*np.exp(-B*t))**(1/v))
    y = A + (K-A)/((C+Q*np.exp(-B*16*(1-(timestep+1)/max_steps)))**(1/v))
    return y

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = batch_count + self.count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (tot_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
