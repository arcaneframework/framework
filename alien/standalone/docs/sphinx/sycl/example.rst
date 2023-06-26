.. _sycl_example:

=======================================
Exemple : how to use Alien SYCL backend
=======================================


Introduction
------------

This tutorial illustrates how to build vectors and matrices, to apply linear algebra operations
and to solve linear systems with solver supporting the Alien SYCL BackEnd.


Using SYCL Linear Algebra
-------------------------

.. code-block:: bash

    /*
     * LINEAR SYSTEM CONSTRUCTION
     */
     
    auto A = Alien::Matrix(mdist);
    auto B = Alien::Vector(vdist);
    auto X = Alien::Vector(vdist);
    auto Y = Alien::Vector(vdist);
    auto R = Alien::Vector(vdist);

    /*
     * Create a SYCL LinearAlgebra object
     */
    Alien::SYCLLinearAlgebra sycl_alg;
    
    /*
     * Apply Blas operation to compute R = B - A*X
     */
    sycl_alg.mult(A,X,Y) ;
    sycl_alg.copy(B,R)
    sycl_alg.axpy(-1,Y,R)
    auto residual = sycl_alg.dot(R,R) ;

Linear Systems resolution
-------------------------

.. code-block:: bash

    /*
     * LINEAR SYSTEM CONSTRUCTION
     */
    auto trace_mng = Alien::Universe.traceMng() ;
     
    auto A = Alien::Matrix(mdist);
    auto B = Alien::Vector(vdist);
    auto X = Alien::Vector(vdist);

    /*
     * Create a SYCLInternalLinearAlgebra instance
     * and get true SYCL Matrix and vectors implementations
     */
    typedef Alien::SYCLInternalLinearAlgebra         AlgebraType ; 
    typedef typename AlgebraType::BackEndType        BackEndType ;
    typedef Alien::Iteration<AlgebraType>            StopCriteriaType ;

    AlgebraType sycl_alg;

    auto const& true_A = A.impl()->get<BackEndType>() ;
    auto const& true_b = b.impl()->get<BackEndType>() ;
    auto&       true_x = x.impl()->get<BackEndType>(true) ;

    /*
     * Create a CG solver and stop criteria
     */
    StopCriteriaType stop_criteria{alg,true_b,tol,max_iteration,output_level>0?trace_mng:nullptr} ;

    typedef Alien::CG<AlgebraType> SolverType ;

    SolverType solver{alg,trace_mng} ;
    solver.setOutputLevel(output_level) ;

    /*
     *Create a Chebyshev polynomial preconditioner
     */
    trace_mng->info()<<"CHEBYSHEV PRECONDITIONER";
    double polynom_factor          = 0.5 ;
    int    polynom_order           = 3 ;
    int    polynom_factor_max_iter = 10 ;

    typedef Alien::ChebyshevPreconditioner<AlgebraType> PrecondType ;
    PrecondType      precond{alg,true_A,polynom_factor,polynom_order,polynom_factor_max_iter,trace_mng} ;
    precond.init() ;

    /*
     * Solve the Linear System A*x=B
     */
    solver.solve(precond,stop_criteria,true_A,true_b,true_x) ;
    
    /*
     * Analyze the solution
     */
    if(stop_criteria.getStatus())
    {
      trace_mng->info()<<"Solver has converged";
      trace_mng->info()<<"Nb iterations  : "<<stop_criteria();
      trace_mng->info()<<"Criteria value : "<<stop_criteria.getValue();
    }
    else
    {
      trace_mng->info()<<"Solver convergence failed";
    }
