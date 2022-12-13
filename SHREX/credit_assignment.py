from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
import copy

gamma_a = 3
gamma_rv = gamma(gamma_a)

class Credit:
    def __init__(self, num_frames, mode='uniform'):
        self.feedbacks = np.zeros((num_frames, 1))
        self.t_prev = 0
        self.t_before = 30
        self.mode = mode

        if self.mode == 'gamma':
            gamma_a = 3
            gamma_rv = gamma(gamma_a)
            self.frames_to_apply = 60

            x = np.linspace(gamma.ppf(0.001, gamma_a),
                gamma.ppf(0.999, gamma_a), self.frames_to_apply)
            self.gamma_pdf = gamma_rv.pdf(x)[::-1]

    def assign(self, feedback, t):
        if self.mode == 'uniform':
            self.feedbacks[max(0,t-self.t_before):t] = feedback
            self.t_prev = t
        
        elif self.mode == 'gamma':
            credit = np.zeros_like(self.feedbacks, dtype=float)

            t_start_credit = max(0, (t - self.frames_to_apply))
            t_start_gamma = max(0, self.frames_to_apply - t)

            credit[t_start_credit:t] = (feedback * self.gamma_pdf[t_start_gamma:])[:,np.newaxis]
            self.feedbacks = self.feedbacks + credit
        
        elif self.mode=='uniform-dense':
            self.feedbacks[self.t_prev:t] = feedback
            self.t_prev = t

        return self.feedbacks

    def get_feedback(self):
        return self.feedbacks

    def plot_feedback(self):
        fig, ax = plt.subplots(1, 1)

        x = np.arange(len(self.feedbacks))
        ax.plot(x, self.feedbacks)
        plt.show()

    # keep only some state-action pairs
    # in future maybe expand this to action space as well?
    def dropout(self, alpha, beta): # alpha is random drop, beta is 0 random drop
        # fbs = np.copy(self.feedbacks)
        # fbs /= np.max(np.abs(fbs))
        # for i, fb in enumerate(fbs):
        #     if np.random.rand() < beta or (fb != 0 and np.random.rand() < abs(fb)):
        #         fbs[i] = 0
        # return fbs
        fbs = np.copy(self.feedbacks)
        # fbs /= np.max(np.abs(fbs))
        zeros=np.where(fbs==0)[0]
        fbs+=1e-8
        pzero = zeros.shape[0]/fbs.shape[0]
        fbs[np.random.choice(np.arange(fbs.shape[0]), int((1-alpha)*fbs.shape[0]), replace=False)]=0
        fbs[np.random.choice(zeros, int(beta*zeros.shape[0]), replace=False)]=0
        keep = fbs!=0
        fbs = fbs[keep]
        fbs -= 1e-8
        return fbs, keep

if __name__ == "__main__":
    # credit = Credit(num_frames=120, mode='uniform')
    credit = Credit(num_frames=1000, mode='uniform-dense')
    credit.assign(feedback=0.75, t=50)
    # credit.plot_feedback()
    credit.assign(feedback=-0.25, t=100)
    # credit.plot_feedback()
    credit.assign(feedback=.5, t=500)
    credit.plot_feedback()
    fbs, keep = credit.dropout(0.9,0.9)
    print(keep.T)

    fig, ax = plt.subplots(1, 1)
    x = np.arange(len(fbs))
    ax.plot(x, fbs)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    x = np.arange(len(keep))
    ax.plot(x, keep)
    plt.show()