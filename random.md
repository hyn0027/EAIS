What SafeDiffuser already do:

给定 diffusion model 其output为$x_0^0, x_1^0, \cdots, x_H^0$，下标为planing traj的标号，上标0意为diffusion的final step，如果我们有一个Lipschitz连续的safety spec $b(x_k^j)\geq 0$, SafeDiffuser 提供了一种方法保证能够满足该safety spec

---

结合我们之前讨论的内容，令$H=0$即我们只有一个输出的control signal，而不是一个traj。SafeDiffuser提供了一种方法，使我们可以保证该模型输出的control signal $x$ 满足  $b(x)\geq 0$，只要$b$是一个Lipschitz连续的函数。

如果我们希望agent学习避免过去的failure，即可以在遇到failure时不断update safety requirement，即 $b$ 函数。

Formally, 如果我们发现一个control signal $x_0$ 会导致failure，则有$b(x_0) < 0$。

但是一个大于或者小于0的information可能用处不大或者信息量太少（？可能只是我的感觉）。如果我们能够尝试给每一个failure point一个评价，类似于这个point在多大程度上错了，是错的很离谱还是slightly wrong，我们也许可以尝试给每个failure point一个评分，类似于 $b(x_0) = k < 0$，可能可以尝试用VLM 评价该control signal的outcome从而给出这个k。

有多个failure point之后，我们可以somehow插值插出来一个approximate b function。应该通过设计一下插值的方法我们可以保证插出来的b是Lipschitz 连续的，至少多项式插值一定可以保证这个

一些可能可以搞点东西的point：

- 也许我们对b的distribution有一个prior假设？
- VLM给failure point评分是否可行/合理
- 怎么插值
- 也许environment有noise，也许我们对某个x是否属于failure point及其可能评分的信息不一定准确

---

- What SafeDiffuser already achieves
  - Given a diffusion model, its output is denoted as: $x_0^0, x_1^0, \cdots, x_H^0$
    where:
    - $H$ represents the total steps of trajectory.
    - The superscript $ 0 $ indicates that the output is from the final step of the diffusion process.
    - The safety specification is defined as a Lipschitz-continuous function $ b(x_k^j) \geq 0 $.
    - SafeDiffuser ensures that the output satisfies this safety condition.
- Application to single-step control signals
  - If we set $ H = 0 $, we only have a single control signal as output instead of a trajectory, SafeDiffuser provides a method to ensure
    $b(x) \geq 0$ given that $ b(x) $ is a Lipschitz-continuous function.
- Learning from past failures and constructing approximate safety functions
  - To enable an agent to learn from past failures:
    - The safety requirement $ b(x) $ can be updated dynamically based on observed failures.
    - Formally, if a control signal $ x' $ leads to a failure, then $b(x') < 0$
  - We may have the potential to provide information about $ b(x_0) $ more than just positive or negative
    - One possibility is to assign a quantitative evaluation to each failure/success point, reflecting the severity of the failure:
    - Assign a score to each failure point $b(x') = k < 0$ where $ k $ represents how badly $ x_0 $ failed.
      - One potential approach: Using VLMs to evaluate the outcome of a control signal and assign a approximate $ k $ score.
  - Given multiple failure points, we may try to approximate the safety functions
    - A simple case: interpolation-based method
    - By designing an appropriate iapproach, we can ensure that the resulting $ b(x) $ function is Lipschitz continuous.
- Potential points for exploration
  - We could have some prior assumption on $ b(x) $
  - Can VLMs reliably score the severity of failures? How?
  - What is the best method for constructing $ b(x) $ from past experiences?
  - The environment might introduce noise. The classification of whether $ x $ is a failure point (and its associated score) might be uncertain or inaccurate.
