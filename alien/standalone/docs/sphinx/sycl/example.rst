.. _sycl_example:

=======================================
Exemple : how to use Alien SYCL backend
=======================================


Introduction
------------

This tutorial illustrates how to build vectors and matrices, to apply linear algebra operations
and to solve linear systems with solver supporting the Alien SYCL BackEnd.

Direct Linear System assembly on device SYCL Handlers
-----------------------------------------------------

.. code-block:: bash

    const Alien::Space s(global_size, "MySpace");
    Alien::MatrixDistribution mdist(s, s, pm);
    Alien::VectorDistribution vdist(s, pm);
    Alien::Matrix A(mdist); // A.setName("A") ;
    Alien::Vector x(vdist); // x.setName("x") ;
    Alien::Vector y(vdist); // y.setName("y") ;

    auto local_size = vdist.localSize();
    auto offset = vdist.offset();

    Alien::SYCLParallelEngine engine;
    {
      auto x_acc = Alien::SYCL::VectorAccessorT<Real>(x);
      engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                    {
                       auto xv = x_acc.view(cgh) ;
                       cgh.parallel_for(engine.maxNumThreads(),
                                        [=](Alien::SYCLParallelEngine::Item<1>::type item)
                                         {
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < local_size; id += item.get_range()[0])
                                               xv[index] = 1.*index;
                                         });

                    }) ;

      auto y_acc = Alien::SYCL::VectorAccessorT<Real>(y);
      engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                    {
                      auto yv = y_acc.view(cgh) ;
                      auto xcv = x_acc.constView(cgh) ;
                      cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < local_size; id += item.get_range()[0])
                                              yv[index] = 2*xcv[index] ;
                                         });
                    }) ;


    {
      Alien::SYCL::MatrixProfiler profiler(A);
      for (Integer i = 0; i < local_size; ++i) {
        Integer row = offset + i;
        profiler.addMatrixEntry(row, row);
        if (row + 1 < global_size)
          profiler.addMatrixEntry(row, row + 1);
        if (row - 1 >= 0)
          profiler.addMatrixEntry(row, row - 1);
     }
    }
    {
      Alien::SYCL::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
      engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                    {
                      auto matrix_acc = builder.view(cgh) ;
                      cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < local_size; id += item.get_range()[0])
                                            {
                                              Integer row = offset + index;
                                              matrix_acc[matrix_acc.entryIndex(row,row)] = 2.;
                                              if (row + 1 < global_size)
                                                matrix_acc[matrix_acc.entryIndex(row, row + 1)] = -1.;
                                              if (row - 1 >= 0)
                                                matrix_acc[matrix_acc.entryIndex(row, row - 1)] = -1.;
                                            }
                                         });
                    }) ;
    }


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
