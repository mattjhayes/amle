Automated Machine Learning Environment (AMLE)
=============================================

AMLE is a simple unopinionated framework for experimenting with
machine learning (ML). I built it to help me learn ML, and
to reduce my workload running ML experiments, by automating repeatable tasks.

It is a perfect example of
`Not Invented, Here <https://en.wikipedia.org/wiki/Not_invented_here>`_
as you can find more performant and fully featured environments elsewhere,
so please consider alternatives like R, or existing Python ML libraries.

The code is still very much under construction.

It is built to the following (aspirational) principles:

* Generic. Just a shim, does not contain ML code, and tries
  to not be opinionated about how ML works or data types
* Reproducibility. Run the same test with same inputs and
  get the same output(s) - or at least statistically similar.
* Reduce experimentation work effort. Support comparative
  testing across different parameters and/or ML algorithms,
  retains historical parameters and results
* Add value to experimentation. Support evolutionary genetic
  approach to configuring algorithm parameters (TBD)
* Visibility. Make it easy to understand how experiments are
  running / ran (TBD)

Documentation may one day be available on `Read the Docs <http://amle.readthedocs.io/en/master/>`_
