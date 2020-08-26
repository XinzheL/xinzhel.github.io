---
layout: post
title:  "Search Algorithms"
date:   2020-04-24 22:21:49 +1000
categories: search
permalink: /:title.html
---

Even though learning techniques become very popular by utilising data to solve problems. Another sub-field is search and optimisation. Many AI problems can be modelled as search problems and then can be solved by search algorithms.

# Different Search Algorithms
The gerneral search algorithms include `uninformed search` and `informed search`. Uninformed search algorithms just utilise information from problems to traverse all the state. Depending on whether duplicated states are visited again, the search topology can be either tree or graph. Actually, the agent just try to traverse all the state in an organised way such as `Breadth First Search` and `Depth First Search` which have been commonly used in many computer science tasks. 

Actually, the essence of search methods are **search strategy**. Therefore, 'intelligence' level also relies on the search strategy. To some extend, I think of uninformed search as 'stupid intelligence' because they do not try to solve problems beyond problem definitions, i.e. not utilise the experience, such as `heuristic` in `informed search` and information of data from `learning` process. 

`Local search algorithms` are much diffferent from previous two since they do not traverse from initial state to find optimal path for goal. They/problems do not care how to find the solution(path & path cost). Interestingly, many  `local search` algorithms tend to use analogies such as `Simulated Annealing`(inspired by metallurgy), `Hill Climbing`(visual description),  `Genetic Algorithm`(inspired by Evolutionary Theory). `Sumulated Annealing` is a method to reduce the possibility of local maximum in `Hill Climbing`.

For understanding the details of algorithms, there are many textbook which gives thorough explanation. [Artificial Intelligence: A Modern Approach](http://aima.cs.berkeley.edu/) is my always-lookup bible for my cohesive understanding of search algorithms. 

In this article, I try to explore some interesting properties of search for efficiency since it determines practicality of algorithms. For efficiency, there are 4 components to evaluate:
* Completeness
* Optimality
* Time Complexity
* Space Complexity

Also, I will explore applications of `local search` algorithms especially how they are integrated into `learning` process.


# Optimality and Completeness(mainly about A*)

Maybe many people hear A* before. However, why it is so popular for solving search problems? The reason is that it ensures optimality and completeness of search. Or in another word, A* is the optimal vesion of A and `Best First Search` algorithms. The relations are described below:

* `Best First Search` algorithms define search strategy via `Heuristc Evaluation Function` f, i.e. This type of algorithm would look at the value of f in the node, f(n), for selection of nodes.  This instances include `greedy best first search` and A/A*.

<!-- Greedy BFS：
Geedy BFS只考虑Heuristic，它不care(存储)之前的路径的Cost信息，所以导致了：一，回不到的有可能更好的path上，因为它压根就不知道。有点类似Backtrack，单通道生物；（Optimality）二，search cost minimal，这算是好的一方面把。其实就是一条路就找到低了，算什么search，直接no plan，走起~ 
再来讲讲它的comleteness的角度，Infinite State就不说了，和DFS一样，很可能踏上就探索宇宙边境的路上了。那Finite State呢？如果是Best First Tree Search，loop~~~ 当一个heuristic比其他的children node小，但其实是图的deadend的时候，因为下次expand它又回去找他妈了(因为和他妈最亲~也就是heuristic最小)，结果他妈又生他一次~loop~loop~ 当然如果是Graph版本的话，graph实现是不会的，因为有个close list存着explored node以便于不会再把这个node放到open list里再让explored。 -->

* In A\A*, `Evaluation Function` f is the summation of `heuristic function` and path cost g, i.e. f(n) = h(n) + g(n). Therefore, the reasonability is the integration of cost from initial to current node and estimated cost from current node to goal.

Let's explain what ensures the completeness and optimality of A*. For completeness, the add of g(n) ensures the completeness which has the same reason as completeness of Uniform Cost Search.

However, for achieving the optimality of path finding and also efficiency, two important properties have to be discussed. Actually, it also increases the efficiency to find the path/goal: 
* Admissible
* Consistency

<!-- 现在看看A*是怎么解决这个问题的？更insightful的说法是怎么通过加了个g(n)解决这个问题的？聪明的朋友又出现了。。。那我是不是可以参考下UCS呢！！！聪明的我当然也想到了，只是我先憋着留下一个Post讲。。。
Completeness:先讲complete，因为也没太多人爱搞它，没有optimality那么惹人爱（嫉妒也没用）。OK. （到此请回顾下为什么Greedy Tree BFS会不complete~）Bingo~ 你对比明白了嘛？在Tree里面现在有了g(n)的加入，你loop几次，f(n)早晚会超过另外的Right node 的f(n_right)。
Optimality：有趣的话题，当然要另开一章BB。之所以有趣，是因为着涉及我们为什么要用启发式算法，要引入heuristic。是因为heuristic能帮我们找到optimal path，而且是更快的，有效率的找到。所以这个话题可以用这样两个问题来开始：
heuristic需要满足什么样性质才能找到Optimal Solution？
进一步的，heuristic需要满足什么样性质才能更快更脑子（memory）的找到Optimal Solution(说白了，就是复杂度问题，暂时没准备在这个问题上涉及太多)？ -->

As said in my bible, 'the tree-search version of A∗ is optimal if h(n) is admissible, while the graph-search version is optimal if h(n) is consistent'. 

The reason why admissibility ensures that the solution of tree search is always optimal is beacause, in any way, it expands all nodes with f(n) < C* where C* is the cost of optimal path. Specifically, all the nodes along optimal path has f(n) lower than cost of optimal path while the f of wrongly chosen path is finally larger than  cost of optimal path sooner or later.

Why graph search cannot ensure optimality with admissibility? Well, it is because it avoids doubl-visiting the nodes. So the better path may be blocked due to this reason.

Here is the example.

![Graph for A*](/assets/img/graph.PNG)


The optimal path is SBAG (total cost=7) but with graph-search, it will generate SAG (total cost=8). The heuristic (see h(B) and h(A) for example) function is admissible (because none of the heuristic overestimates the actual cost) but inconsistent (because h(B) > c(A,B) + h(A) instead of h(B) <= c(A, B) + h(A), the so-called triangle inequality).

If we use graph-search for A* algorithm here, after expanding S (put S in closed list), we will generate A (f=4+1) and B(f=2+5). So we will expand A (put A in closed list) next which generates G(f=8+0). Then B will expand and generate A(f=3+1), however, we already see A in the closed list (which assumes we already visited A with lower cost). Therefore, only G left in the frontier/open.

But with tree-search, we don't have the problem above as we treat all generated nodes a unique node. For example, the first time we generate A, we will use SA; and the second time, we will use SBA. They represent two different nodes. Hence, an admissible heuristic guarantees an optimal solution in tree search but requires a consistent heuristic in graph-search.



# Model the problems for search
Actually, for problem solving as search in the real world, the most challenging part is not algorithms themselves but how to model problems which can be solved as search. There are many toy games given as examples in the textbook, such as Maze, 8 puzzle(find the shortest path) and Soduku, 8 queens(optimization of configuration).


<!-- Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->


<!-- http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#breaking-ties -->

<!-- https://movingai.com/astar.html -->