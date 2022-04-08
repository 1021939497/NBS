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
    '''
    Returns the components of an undirected graph specified by the binary and
    undirected adjacency matrix adj. Components and their constitutent nodes
    are assigned the same index and stored in the vector, comps. The vector,
    comp_sizes, contains the number of nodes beloning to each component.
    Parameters
    ----------
    A : NxN np.ndarray
        binary undirected adjacency matrix
    no_depend : Any
        Does nothing, included for backwards compatibility
    Returns
    -------
    comps : Nx1 np.ndarray
        vector of component assignments for each node
        
    comp_sizes : Mx1 np.ndarray
        vector of component sizes
    Notes
    -----
    Note: disconnected nodes will appear as components with a component
    size of 1
    Note: The identity of each component (i.e. its numerical value in the
    result) is not guaranteed to be identical the value returned in BCT,
    matlab code, although the component topology is.
    Many thanks to Nick Cullen for providing this implementation
    '''
    if not np.all(A == A.T):  # ensure matrix is undirected
        raise BCTParamError('get_components can only be computed for undirected'
                            ' matrices.  If your matrix is noisy, correct it with np.around')

    A = binarize(A, copy=True)#二值化
    n = len(A)
    np.fill_diagonal(A, 1)#用1填充A的对角线

    edge_map = [{u,v} for u in range(n) for v in range(n) if A[u,v] == 1]#A中为1的位置
    
    union_sets = []

    
#将相互关联的脑区放在一起构成一个组件放入union_sets
    for item in edge_map:
        temp = []
        for s in union_sets:
            #print('s',s)
            if not s.isdisjoint(item):#判断两个集合s和item是否包含相同的元素，如果没有返回 True，否则返回 False
                #print(item)
                item = s.union(item)#合并s和item
                #print('s.union(item)后的item',item)
            else:
                temp.append(s)#列表末尾添加新的对象。
        temp.append(item)
        
        union_sets = temp
    #print('union_sets:',union_sets) 
#union_sets中会包含多个脑区组件
    
    #comps为每个脑区 所属组件的索引编号（从1开始）
    comps = np.array([i+1 for v in range(n) for i in
        range(len(union_sets)) if v in union_sets[i]])
    #comp_sizes为每个组件的大小
    comp_sizes = np.array([len(s) for s in union_sets])

    return comps, comp_sizes


@due.dcite(BibTeX(ZALESKY2010), description="Network-based statistic")
def nbs_bct(x, y, thresh, k=1000, tail='both', paired=False, verbose=False, seed=None):
    '''
    Performs the NBS for populations X and Y for a t-statistic threshold of
    alpha.
    Parameters
    ----------
    x : NxNxP np.ndarray
        matrix representing the first population with P subjects. must be
        symmetric.
    y : NxNxQ np.ndarray
        matrix representing the second population with Q subjects. Q need not
        equal P. must be symmetric.
    thresh : float
        minimum t-value used as threshold
    k : int
        number of permutations used to estimate the empirical null
        distribution
    tail : {'left', 'right', 'both'}
        enables specification of particular alternative hypothesis
        'left' : mean population of X < mean population of Y
        'right' : mean population of Y < mean population of X
        'both' : means are unequal (default)
    paired : bool
        use paired sample t-test instead of population t-test. requires both
        subject populations to have equal N. default value = False
    verbose : bool
        print some extra information each iteration. defaults value = False
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    pval : Cx1 np.ndarray
        A vector of corrected p-values for each component of the networks
        identified. If at least one p-value is less than alpha, the omnibus
        null hypothesis can be rejected at alpha significance. The null
        hypothesis is that the value of the connectivity from each edge has
        equal mean across the two populations.
    adj : IxIxC np.ndarray
        an adjacency matrix identifying the edges comprising each component.
        edges are assigned indexed values.
    null : Kx1 np.ndarray
        A vector of K sampled from the null distribution of maximal component
        size.
    Notes
    -----
    ALGORITHM DESCRIPTION
    The NBS is a nonparametric statistical test used to isolate the
    components of an N x N undirected connectivity matrix that differ
    significantly between two distinct populations. Each element of the
    connectivity matrix stores a connectivity value and each member of
    the two populations possesses a distinct connectivity matrix. A
    component of a connectivity matrix is defined as a set of
    interconnected edges.
    The NBS is essentially a procedure to control the family-wise error
    rate, in the weak sense, when the null hypothesis is tested
    independently at each of the N(N-1)/2 edges comprising the undirected
    connectivity matrix. The NBS can provide greater statistical power
    than conventional procedures for controlling the family-wise error
    rate, such as the false discovery rate, if the set of edges at which
    the null hypothesis is rejected constitues a large component or
    components.
    The NBS comprises fours steps:
    1. Perform a two-sample T-test at each edge indepedently to test the
       hypothesis that the value of connectivity between the two
       populations come from distributions with equal means.
    2. Threshold the T-statistic available at each edge to form a set of
       suprathreshold edges.
    3. Identify any components in the adjacency matrix defined by the set
       of suprathreshold edges. These are referred to as observed
       components. Compute the size of each observed component
       identified; that is, the number of edges it comprises.
    4. Repeat K times steps 1-3, each time randomly permuting members of
       the two populations and storing the size of the largest component
       identified for each permuation. This yields an empirical estimate
       of the null distribution of maximal component size. A corrected
       p-value for each observed component is then calculated using this
       null distribution.
    [1] Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic:
        Identifying differences in brain networks. NeuroImage.
        10.1016/j.neuroimage.2010.06.041
    '''
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
        '''
        n = len(A - B)
        df = n - 1
        sample_ss = np.sum((A - B)**2) - np.sum(A - B)**2 / n
        print('sample_ss=',sample_ss)
        unbiased_std = np.sqrt(sample_ss / (n - 1))
        print('unbiased_std=',unbiased_std)
        z = np.mean(A - B) / unbiased_std
        print('z=',z)
        t = z * np.sqrt(n)
        '''
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

    # only consider upper triangular edges
    #只考虑上三角边
    aa = np.ones((n, n))
    #np.where返回两个数组，恰好为符合要求的坐标，第一个为坐标行数（y坐标），第二个为坐标列数（x坐标）
    ixes = np.where(np.triu(aa, 1))#np.triu第二个参数为1则返回对角线上三角（不包含对角）
    #ixes包含了np.where返回的两个坐标数组

    # number of edges
    #m为ixes二维矩阵的列数，即n*n二维矩阵上三角的元素数
    m = np.size(ixes, axis=1)

    # vectorize connectivity matrices for speed
    ##m为（x,y）二维的上三角边数，nx，ny为z坐标上的数目
    xmat, ymat = np.zeros((m, nx)), np.zeros((m, ny))

    for i in range(nx):
        xmat[:, i] = x[:, :, i][ixes].squeeze()
        #ixes包含的行列坐标为对角线上三角的每个位置
        #根据ixes包含的行列坐标将三维（x,y,z）中z个二维（x，y）按照对应ixes坐标提取（x，y）
        #按行优先变成一维数组存入xmat第一维，一共有z个
    for i in range(ny):
        ymat[:, i] = y[:, :, i][ixes].squeeze()
    del x, y#del语句默认将变量声明为本地变量


    # perform t-test at each edge
    t_stat = np.zeros((m,))#为每个元素算xmat和ymat的t-test
    p = np.zeros((m,))
    
    for i in range(m):
        if paired:
            t_stat[i],p[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
        else:
            t_stat[i],p[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
    
    thresh = 0.001
    ind_t = np.where(p < thresh)#p小于阈值的位置坐标

    if len(ind_t) == 0:
        raise BCTParamError("Unsuitable threshold")

    # suprathreshold adjacency matrix#超阈值n*n连接矩阵
    adj = np.zeros((n, n))
    adj[(ixes[0][ind_t], ixes[1][ind_t])] = 1
    # adj[ixes][ind_t]=1
    adj = adj + adj.T#把下三角和上三角对称出来

    a, sz = get_components(adj)
    #comps中存入a
    #comp_sizes存入sz

    # convert size from nodes to number of edges
    # only consider components comprising more than one node (e.g. a/l 1 edge)
    #sz>1 仅考虑包含节点数>1的组件
    #ind_sz为满足节点数>1的组件的坐标索引
    ind_sz, = np.where(sz > 1)#
    #ind_sz存满足组件大小>1的组件编号（+1是为了编号从1开始）
    ind_sz += 1 
    
    #返回矩阵ind_sz的元素个数，即大小符合要求的组件个数
    nr_components = np.size(ind_sz)#返回矩阵ind_sz的元素个数,就是组件大小>1的组件数目
    sz_links = np.zeros((nr_components,))
    #print('nr_components',nr_components)
    for i in range(nr_components):
        nodes, = np.where(ind_sz[i] == a)# 属于编号为ind_sz[i]的组件的 脑区索引  node就是若干脑区索引组成的列表
        #print(nodes)
        #np.ix_笛卡尔积，把每个脑区和其他的脑区 两两罗列
        #笛卡尔积：(1,2)×(1,2) = (1,1),(1,2),(2,1),(2,2)
        #在adj（超阈值矩阵）找到这两个脑区即adj（i,j）的总和再除2   除2因为是非定向，（i，j）和（j，i）一样
        sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
        #np.ix_函数就是输入两个数组，产生笛卡尔积的映射关系
        adj[np.ix_(nodes, nodes)] *= (i + 2)#把组件大小>1的组件中脑区对应的adj（i,j）=1放大成2
    #print(adj)#输出调整好后的adj(把组件大小>1的组件所包含的脑区笛卡尔积得到对应位置（i,j）的adj放大)
    #组件大小=1的adj在上面没进行操作，要把这些大小为1的组件删掉，adj对应的位置也要从1变为0
    #subtract 1 to delete any edges not comprising a component
    #adj为正数的都-1，放大成2的就变成1；没放大的1就变为0，达到删除大小为1的组件的目的
    adj[np.where(adj)] -= 1
    #print(adj)


    #print(sz_links)
    if np.size(sz_links):
        max_sz = np.max(sz_links)
    else:
        # max_sz=0
        raise BCTParamError('True matrix is degenerate')
    print('max component size is %i' % max_sz)


    # estimate empirical null distribution of maximum component size by
    # generating k independent permutations
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
        
        #pvals[i]为k次排列中得到的最大的组件大小 超过原第i个组件中涉及脑区连接数的次数/排列检验次数
        #sz_link[i]存第i个组件中涉及的脑区连接数
        pvals[i] = np.size(np.where(null >= sz_links[i])) / k

    return pvals, adj, null
