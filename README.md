# INCREASING BOTH BATCH SIZE AND LEARNING RATE ACCELERATES STOCHASTIC GRADIENT DESCENT
# Abstract
The performance of mini-batch stochastic gradient descent (SGD) strongly de-
pends on the setting of batch size and learning rate to minimize the empirical
loss in training a deep neural network. In this paper, we give theoretical analy-
ses of mini-batch SGD with four schedulers, (i) constant batch size and decay-
ing learning rate scheduler, (ii) increasing batch size and decaying learning rate
scheduler, (iii) increasing batch size and increasing learning rate scheduler, and
(iv) increasing batch size and warmup decaying learning rate scheduler. We show
that mini-batch SGD using scheduler (i) does not always minimize the expectation
of the full gradient norm of the empirical loss, while mini-batch SGD using each
of schedulers (ii), (iii), and (iv) minimizes it. In particular, using scheduler (iii)
and (iv) accelerates mini-batch SGD. We provide numerical results supporting the
analyses such that using schedulers (iii) and (iv) minimizes the full gradient norm
of the empirical loss faster than using other schedulers.
