# -*- coding: utf-8 -*-
#
# BAYES LOOP - DATA ANALYZER (v1.0)
# ---------------------------------
# contact: christoph.mark@fau.de
#
# This software is distributed under the MIT License (MIT)
#
# Copyright (c) 2015 Christoph W. Mark
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


# cut the GUI out of the provided code
#
class BayesLoop(object):
    def __init__(self):
        self.save_plots = False
        # algorithm parameters        
        self.gridSize = 200
        self.boundaries = [0.0,3.0,-1.5,1.5]
        self._aBound = self.boundaries[0:2]
        self._qBound = self.boundaries[2:4]
        self.use_reverse = True
        # why are the boundaries not [-1,1]?
        self.pMin = 10**(-7)
        self.Ra = 2
        self.Rq = 2
        self.kernel_on = True
        self.data = np.array([])
        self.progressIndicator = 0
    
    def set_aBound(self, val):
        assert(len(val) == 2)
        self._aBound = val
        self.boundaries[0:2] = val
    def get_aBound(self):
        return self._aBound

    def set_qBound(self, val):
        assert(len(val) == 2)
        self.boundaries[2:4] = val
        self._qBound = val
    def get_qBound(self):
        return self._qBound

    aBound = property(get_aBound, set_aBound)
    qBound = property(get_qBound, set_qBound)

    def get_R(self):
        # compute the actual R values
        amin, amax, qmin, qmax = self.boundaries
        return (self.Rq * (qmax-qmin)/self.gridSize,
            self.Ra * (amax-amin)/self.gridSize)
        
        
    # Bayesian Inference Algorithm
    # ----------------------------
    # For further documentation, see simulatedExamples.py.
    def compLike(self, vp, v):
        return np.exp(-((v[0] - self.qGrid*vp[0])**2 + (v[1] - self.qGrid*vp[1])**2)/(2*self.a2Grid) - np.log(2*np.pi*self.a2Grid))
        
    def compNewPrior(self, oldPrior, like):
        post = oldPrior*like 
        post /= np.sum(post) # why do we normalise here and not at the en
        
        newPrior = post
        
        mask = newPrior < self.pMin
        newPrior[mask] = self.pMin
        
        if self.kernel_on:
            ker = np.ones((2*self.Rq + 1, 2*self.Ra+1))/((2*self.Rq+1)*(2*self.Ra+1))
            newPrior = convolve2d(newPrior, ker, mode='same', boundary='symm')
        return newPrior
        
    def compPostSequ(self, uList, reverse=True):
        dist = np.empty((len(uList),self.gridSize,self.gridSize))
        dist[0].fill(1.0/(self.gridSize**2))
    
        for i in np.arange(1,len(uList)):
            dist[i] = self.compNewPrior(dist[i-1], self.compLike(uList[i-1], uList[i]))
            self.progressIndicator += 1
            if not reverse:
                dist[i] /= np.sum(dist[i])
        if not reverse:
            return dist[1:]
    
        backwardPrior = np.ones((self.gridSize,self.gridSize))/(self.gridSize**2)
        for i in np.arange(1,len(uList))[::-1]:
            like = self.compLike(uList[i-1], uList[i])
        
            dist[i] = dist[i-1]*like*backwardPrior
            dist[i] /= np.sum(dist[i])
            
            backwardPrior = self.compNewPrior(backwardPrior, self.compLike(uList[i-1], uList[i]))
            self.progressIndicator += 1
        
        return dist[1:]
        
    def compPostMean(self, postSequ):
        qMean = [np.sum(post*self.qGrid) for post in postSequ]
        aMean = [np.sum(post*self.aGrid) for post in postSequ]
        
        return np.array([qMean,aMean])
    
    # Plot Routines
    # -------------
    # The following functions allow to plot the reconstructed posterior mean
    # values along with the corresponding marginal distributions as well as 
    # the time-averaged joint posterior distribution
    def plotPosteriorSequence(self, postSequ, postMean):
        plt.subplot(121)
        margPostQ = np.sum(postSequ, axis=2).T
        
        plt.imshow(margPostQ,
                   origin=0,
                   cmap='Blues',
                   extent=[1]+[postSequ.shape[0]]+self.boundaries[2:4],
                   aspect=.8*postSequ.shape[0]/(self.boundaries[3]-self.boundaries[2]))
               
        plt.plot(np.arange(1, postSequ.shape[0]+1), postMean[0], c='k', lw=.75)
        plt.xlim((1,postSequ.shape[0]))
        plt.ylim(self.boundaries[2:4])
        plt.title('persistence')
        plt.xlabel('time step')
        
        plt.subplot(122)
        margPostA = np.sum(postSequ, axis=1).T
        
        plt.imshow(margPostA,
                   origin=0,
                   cmap='Reds',
                   extent=[1]+[postSequ.shape[0]]+self.boundaries[0:2],
                   aspect=.8*postSequ.shape[0]/(self.boundaries[1]-self.boundaries[0]))
        plt.plot(np.arange(1, postSequ.shape[0]+1), postMean[1], c='k', lw=.75)
        plt.xlim((1,postSequ.shape[0]))
        plt.ylim(self.boundaries[0:2])
        plt.title('activity')
        plt.xlabel('time step')
        
        plt.tight_layout()
    
    def plotAveragePosterior(self, avgPost):
        qValues = np.linspace(self.qBound[0], self.qBound[1], self.gridSize+2)[1:-1]
        aValues = np.linspace(self.aBound[0], self.aBound[1], self.gridSize+2)[1:-1]
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.contourf(qValues,aValues,avgPost.T, cmap='Oranges', zorder=0)
        ax.contour(qValues,aValues,avgPost.T, colors='k', linewidths=.5, zorder=1)
        
        ax.set_xlabel('persistence')
        ax.set_ylabel('activity')
    
    # Data Import
    # -----------
    # The software loads txt-Files using numpy's loadtxt function. Only data
    # formatted in two columns separated by whitespaces is currently supported.
    def loadFile(self):
        # load two columns of data representing a track using numpy loadtxt into self.data
        pass
   
        
    def log(self, string):
        print(string)

    def save(self, name_form):
        name = name_form.format('postMean')
        print('saving to name', name)
        np.save(name, self.postMean)
        name = name_form.format('avgPost')
        print('saving to name', name)
        np.save(name, self.avgPost)

    # Analysis
    # -----------
    # The paper also explains how to use maximum likelihood estimation
    # This is probably the more natural way to get (q,a) estimates for crawling data
    # since we want to assume that the crawling behaviour doesn't change 
    def q_estimator(self, sample=None):
        u = self.data
        # can be simplified by summing over both axes
        if sample is None:
            sample = np.array(range(u.shape[0]-1))
        return np.sum((u[1:]*u[:-1]).sum(axis=1)[sample])/np.sum((u[:-1]*u[:-1]).sum(axis=1)[sample])

    def MLE(self):
        u = self.data
        qhat = self.q_estimator()
        upart = u[1:] - qhat*u[:-1]
        return qhat, np.sqrt(np.sum(upart*upart)/(2*(u.shape[0]-1)))

    def startAnalysis(self):
        # build parameter grid
        self.log('build parameter grid...')
        self.qGrid  = (np.array([np.linspace(self.qBound[0], self.qBound[1], self.gridSize+2)[1:-1]]*self.gridSize)).T
        self.aGrid  = (np.array([np.linspace(self.aBound[0], self.aBound[1], self.gridSize+2)[1:-1]]*self.gridSize))
        self.a2Grid = (np.array([np.linspace(self.aBound[0], self.aBound[1], self.gridSize+2)[1:-1]]*self.gridSize))**2    

        if self.use_reverse:
            self.log('Computing posterior sequence in both directions...')
        else:
            self.log('Computing posterior sequence...')
        
        # compute posterior sequence        
        postSequ = self.compPostSequ(self.data, reverse=self.use_reverse)
        self.postSequ = postSequ
       
        self.log('Computing posterior mean values...')
        
        # compute posterior mean values (shape (2,n))
        postMean = self.compPostMean(postSequ)
        self.postMean = postMean

        avgPost = np.sum(postSequ, axis=0)
        avgPost /= np.sum(avgPost)
        self.avgPost = avgPost # shape (gridsize, gridsize)

        self.log('Finished Analysis...')
        
        if self.save_plots:
            self.log('Saving results...')

        
            # do some plotting & saving        
            self.plotPosteriorSequence(postSequ, postMean)
            plt.savefig('posteriorSequence.pdf', format='pdf')
            plt.close('all')
            
            
            self.plotAveragePosterior(avgPost)
            plt.savefig('averagePosterior.pdf', format='pdf')
            plt.close('all')
            
            # save analysis data        
            np.savetxt('posteriorMeanValues.txt', postMean.T) 
            np.savetxt('qGrid.txt', self.qGrid)
            np.savetxt('aGrid.txt', self.aGrid)
            np.savetxt('averagePosterior.txt', avgPost)
            
        self.progressIndicator = 0
        
# Main Routine
# ------------
if __name__ == '__main__':

    bayesloop = BayesLoop()
