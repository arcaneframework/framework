.. _user_concepts:

=========================
Alien's high level design
=========================

Alien's design key objective is to provide a robust interface for linear algebra users in the context of distributed
memory computations.

Robustness is achieved relying on well defined concepts to ensure a mathematically correct code, as well as a valid
parallel code.

Mathematical Choices
====================

We will present quickly important mathematical aspects of Alien. First we will introduce the scope of Alien, and then we will
describe more precisely how Alien uses mathematical objects such as matrices.

Linear Algebra
--------------

Alien focuses on linear algebra, dealing with matrices and vectors. We do not support nonlinear solvers,
such as non-linear Newton algorithms.

Alien provides access to solvers for linear equations: for given matrix :math:`A` and right hand side vector :math:`b`,
find vector :math:`x` which satisfies :eq:`linear_eq`.

.. math:: A x = b
   :label: linear_eq

Alien also provides access to linear algebra basic computations such as matrix-vector products, linear combinations, ...

Spaces
------

In Alien, we chose to view linear algebra objects as defined on generic indexed *spaces*.

Indeed, there are several ways to define a matrix, and we retain the following:
given three sets :math:`(I,J,K)`, a matrix :math:`M` is a family of elements of :math:`K`,
indexed by the cartesian product :math:`I \times J`; i.e. an application :math:`M : I \times J \mapsto K`, see
[bourbaki_matrix]_ for more details.

Most common indexed space are :math:`\{0,..,n-1\}` or :math:`\{1,..,n\}`.

Using named spaces allow us to ensure strict compatibility checks when computing.
For example, if :math:`u_I` and :math:`u_J` are two vectors defined on :math:`I` and :math:`J`,
we can detect that computing matrix-vector product :math:`M . u_I` is invalid while :math:`M . u_J` is valid.

If :math:`I` and :math:`J` have the same size, as it is the case for square matrices, it is impossible to distinguish
mathematically valid operations from invalid without using set's names.

Another advantage of using these named spaces is that we can unambiguously define sub-matrices.

Trilinos [trilinos]_ uses something similar but less formalized with its `maps <https://docs.trilinos.org/dev/packages/epetra/doc/html/classEpetra__Map.html>`_.

Software design
===============

Multiple representations
------------------------

Alien is mainly a wrapper over external linear algebra libraries. The idea is to not reimplement linear solvers.
However, we focus on interoperability between libraries and functionalities.

Alien's main idea is to allow dynamic (runtime) change of linear solver implementation. For example, one can try to solve
a linear system using a fast but not numerically robust algorithm, but yet be able to switch to a slow and robust one
if the former has failed.

Alien will convert data structures between libraries and deal with the distributed memory aspects.


Stateless objects
-----------------

To ensure easier of Alien, as well as more robust code, we have chosen to not rely on state for our objects.
That means that we do not want the user to deal with notions liked, is my matrix correctly finalized before computing with it ?

To enable that idea, we heavily rely on a converter pattern. That means we will have a lot of specific objects, that can only
perform one task, such as filling in or compute linear expression. To perform different operations, user has to
*explicitly* convert between types.


Coherent APIs
-------------


References
==========

.. [bourbaki_matrix]  N. Bourbaki, Algèbre, Chapitres 1 à 3, Springer, 2006, 2e éd.
.. [trilinos] The Trilinos Project Website, https://trilinos.github.io
