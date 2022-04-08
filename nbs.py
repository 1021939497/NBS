from __future__ import division, print_function
import numpy as np

from bct.utils import BCTParamError, get_rng, binarize
from bct.algorithms import get_components
from bct.due import due, BibTeX
from bct.citations import ZALESKY2010
import scipy.stats as stats

# FIXME considerable gains could be realized using vectorization, although
# generating the null distribution will take a while
def get_components(A, no_depend=False):
    if not np.all(A == A.T):  # ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
                            ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)
    n = len(A)
    np.fill_diagonal(A, 1)
    edge_map = [{u,v} for u in range(n) for v in range(n) if A[u,v] == 1]#A中为1的位置
    union_sets = []
    for item in edge_map:
        temp = []
        for s in union_sets:
            if not s.isdisjoint(item):
                item = s.union(item)
            else:
                temp.append(s)
        temp.append(item)
        union_sets = temp
    comps = np.array([i+1 for v in range(n) for i in
        range(len(union_sets)) if v in union_sets[i]])
    comp_sizes = np.array([len(s) for s in union_sets])
    return comps, comp_sizes

def nbs_bct(x, y, thresh, k=1000, tail='both', paired=False, verbose=False, seed=None):
    rng = get_rng(seed)
    def ttest2_stat_only(x, y, tail):
        t = np.mean(x) - np.mean(y)
        print('t=',t)
        n1, n2 = len(x), len(y)
        s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                     * np.var(y, ddof=1)) / (n1 + n2 - 2))
        denom = s * np.sqrt(1 / n1 + 1 / n2)
        print('denom=',denom)
        if denom == 0:
            return 0
        if tail == 'both':
            return np.abs(t / denom)
        if tail == 'left':
            return -t / denom
        else:
            return t / denom
        
    def ttest_paired_stat_only(A, B, tail):#配对t-test
        t,p = stats.ttest_rel(A, B)
        if tail == 'both':
            return np.abs(t),p
        if tail == 'left':
            return -t,p
        else:
            return t,p

    if tail not in ('both', 'left', 'right'):
        raise BCTParamError('Tail must be both, left, right')

    ix, jx, nx = x.shape
    iy, jy, ny = y.shape

    if not ix == jx == iy == jy:
        raise BCTParamError('Population matrices are of inconsistent size')
    else:
        n = ix

    if paired and nx != ny:
        raise BCTParamError('Population matrices must be an equal size')

   
    aa = np.ones((n, n))
    
    ixes = np.where(np.triu(aa, 1))
    
    m = np.size(ixes, axis=1)

   
    xmat, ymat = np.zeros((m, nx)), np.zeros((m, ny))

    for i in range(nx):
        xmat[:, i] = x[:, :, i][ixes].squeeze()
        
    for i in range(ny):
        ymat[:, i] = y[:, :, i][ixes].squeeze()
    del x, y


    
    t_stat = np.zeros((m,))
    p = np.zeros((m,))
    
    for i in range(m):
        if paired:
            t_stat[i],p[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
        else:
            t_stat[i],p[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
    
    thresh = 0.001
    ind_t = np.where(p < thresh)

    if len(ind_t) == 0:
        raise BCTParamError("Unsuitable threshold")


    adj = np.zeros((n, n))
    adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    adj = adj + adj.T#把下三角和上三角对称出来

    a, sz = get_components(adj)
    
    ind_sz, = np.where(sz > 1)
    ind_sz += 1 
   
    nr_components = np.size(ind_sz)
    sz_links = np.zeros((nr_components,))
    #print('nr_components',nr_components)
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)
       
        sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
       
        adj[np.ix_(nodes, nodes)] *= (i + 2)
   
    adj[np.where(adj)] -= 1
       if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        raise BCTParamError('True matrix is degenerate')
    print('max component size is %i' % max_sz)

    # k为排列检验次数
    print('estimating null distribution with %i permutations' % k)

    null = np.zeros((k,))
    
    hit = 0
    max_size = []
    max_size.append(max_sz)
    for u in range(k):
        # randomize
        if paired:
            indperm = np.sign(0.5 - rng.rand(1, nx))
            d = np.hstack((xmat, ymat)) * np.hstack((indperm, indperm))
        else:
            d = np.hstack((xmat, ymat))[:, rng.permutation(nx + ny)]

        t_stat_perm = np.zeros((m,))
        p_perm = np.zeros((m,))
        for i in range(m):
            if paired:
                t_stat_perm[i],p_perm[i] = ttest_paired_stat_only(d[i, :nx], d[i, -nx:], tail)
            else:
                t_stat_perm[i] = ttest2_stat_only(d[i, :nx], d[i, -ny:], tail)

        ind_t, = np.where(p_perm < thresh)

        adj_perm = np.zeros((n, n))
        adj_perm[(ixes[0][ind_t], ixes[1][ind_t])] = 1
        adj_perm = adj_perm + adj_perm.T

        a, sz = get_components(adj_perm)

        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components_perm = np.size(ind_sz) #满足组件大小>1的组件个数
        
        sz_links_perm = np.zeros((nr_components_perm))
        
        for i in range(nr_components_perm):
            
            nodes, = np.where(ind_sz[i] == a)
            
            
            #adj总和/2  除2因为是非定向，（i，j）和（j，i）一样
            sz_links_perm[i] = np.sum(adj_perm[np.ix_(nodes, nodes)]) / 2

        if np.size(sz_links_perm):#sz_links_perm不为空
            null[u] = np.max(sz_links_perm) #当前排列检验中最大的组件大小
            
        else:
            null[u] = 0

        # compare to the true dataset
        if null[u] >= max_sz: #max_sz为进行排列检验之前得到的最大的组件大小
            hit += 1 #超过之前得到的最大的组件大小 的次数
            #记录比max_sz大的那几个组件大小
            max_size.append(null[u])
            
        
        #此处输出的p-value为超过初始最大组件大小的次数/当前排列检验次数
        if verbose:#是否打印排列检验信息
            print(('permutation %i of %i.  Permutation max is %s.  Observed max'
                   ' is %s.  P-val estimate is %.3f') % (
                u, k, null[u], max_sz, hit / (u + 1)))
        
        elif (u % (k / 10) == 0 or u == k - 1):
            print('permutation %i of %i.  p-value so far is %.3f' % (u, k,
                                                                     hit / (u + 1)))
    
    #返回仅包含最大组件内脑区的adj
    
    
    
    pvals = np.zeros((nr_components,))
    # calculate p-vals
    for i in range(nr_components):
       
        pvals[i] = np.size(np.where(null >= sz_links[i])) / k

    return pvals, adj, null
