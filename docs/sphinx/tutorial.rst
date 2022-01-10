Tutorial
========


Intro
-----

This tutorial illustrates how to build a linear system and solve it with various linear solver algorithm.

We consider the Laplacian problem on a 2D square mesh of size :math:`N_X \times N_Y`. Unknowns are related to the mesh nodes 
indexed by :math:`(i,j)`. We use a 5-Points stencil to discretize the problem.

First of all Alien provides tools to initialize the library, the MPI parallel environment to parametrize IO.
The following code illustrates how to initialize :

- the ParallelMng object

- the TraceMng object


.. code-block:: bash
    
    // INITIALIZE PARALLEL ENVIRONMENT
    Environment::initialize(argc, argv);

    auto parallel_mng = Environment::parallelMng();
    auto trace_mng = Environment::traceMng();

    auto comm_size = Environment::parallelMng()->commSize();
    auto comm_rank = Environment::parallelMng()->commRank();

    trace_mng->info() << "NB PROC = " << comm_size;
    trace_mng->info() << "RANK    = " << comm_rank;

    Arccore::StringBuilder filename("tutorial.log");
    Arccore::ReferenceCounter<Arccore::ITraceStream> ofile;
    if (comm_size > 1) {
      filename += comm_rank;
      ofile = Arccore::ITraceStream::createFileStream(filename.toString());
      trace_mng->setRedirectStream(ofile.get());
    }
    trace_mng->finishInitialize();

    Alien::setTraceMng(trace_mng);
    Alien::setVerbosityLevel(Alien::Verbosity::Debug);

Space
-----

The Space concept enable to modelize the mathematical algebraic real space :math:`R^N` of dimension :math:`N`

To build this concept several tools are provided:

- the `IndexeManager` package provides helper tools to manage `Integer IndexSets`

- the `Distribution` package provides helper tools to manage the partition of `IndexSets` between MPI processes

.. code-block:: bash

    int Nx = 10;
    int Ny = 10;

    /*
     * MESH PARTITION ALONG Y AXIS
     *
     */
    int local_ny = Ny / comm_size;
    int r = Ny % comm_size;

    std::vector<int> y_offset(comm_size + 1);
    y_offset[0] = 0;
    for (int ip = 0; ip < r; ++ip)
      y_offset[ip + 1] = y_offset[ip] + local_ny + 1;

    for (int ip = r; ip < comm_size; ++ip)
      y_offset[ip + 1] = y_offset[ip] + local_ny;

    // Define a lambda function to compute node unique ids from the 2D (i,j) coordinates
    // (i,j) -> uid = node_uid(i,j)
    auto node_uid = [&](int i, int j) { return j * Nx + i; };

    /*
     * DEFINITION of Unknowns Unique Ids and  Local Ids
     */
    Alien::UniqueArray<UID> uid;
    Alien::UniqueArray<LID> lid;
    int first_j = y_offset[comm_rank];
    int last_j = y_offset[comm_rank + 1];

    int index = 0;
    for (int j = first_j; j < last_j; ++j) {
      for (int i = 0; i < Nx; ++i) {
        uid.add(node_uid(i, j));
        lid.add(index);
        ++index;
      }
    }

    /*
     * DEFINITION of an abstract family of unknowns
     */
    Alien::DefaultAbstractFamily family(uid, parallel_mng);
    Alien::IndexManager index_manager(parallel_mng);

    /*
     * Creation of a set of indexes
     */
    auto indexSetU = index_manager.buildScalarIndexSet("U", lid, family, 0);

    // Combine all index set and create Linear system index system
    index_manager.prepare();

    auto global_size = index_manager.globalSize();
    auto local_size = index_manager.localSize();

    trace_mng->info() << "GLOBAL SIZE : " << global_size;
    trace_mng->info() << "LOCAL SIZE  : " << local_size;

    /*
     * DEFINITION of
     * - Alien Space,
     * - matrix and vector distributions
     * to manage the distribution of indexes between all MPI processes
     */

    auto space = Alien::Space(global_size, "MySpace");

    auto mdist =
    Alien::MatrixDistribution(global_size, global_size, local_size, parallel_mng);
    auto vdist = Alien::VectorDistribution(global_size, local_size, parallel_mng);

    trace_mng->info() << "MATRIX DISTRIBUTION INFO";
    trace_mng->info() << "GLOBAL ROW SIZE : " << mdist.globalRowSize();
    trace_mng->info() << "LOCAL ROW SIZE  : " << mdist.localRowSize();
    trace_mng->info() << "GLOBAL COL SIZE : " << mdist.globalColSize();
    trace_mng->info() << "LOCAL COL SIZE  : " << mdist.localColSize();

    trace_mng->info() << "VECTOR DISTRIBUTION INFO";
    trace_mng->info() << "GLOBAL SIZE : " << vdist.globalSize();
    trace_mng->info() << "LOCAL SIZE  : " << vdist.localSize();


Matrix
------

The Matrix concept represents a set of :math:`N_i` linear equations (rows) :math:`(y_i)` of :math:`N_j` unknowns :math:`(x_j)` (columns). 
This represents a linear application :math:`S_X \mapsto S_Y` with :math:`x \in S_X`, :math:`y \in S_Y` and :math:`x \mapsto y=A*x`. 
Usually the dimension of :math:`S_X` and :math:`S_Y` are equal, :math:`N_i=N_j`. In that case the matrix is square.

.. code-block:: bash

    /*
     * MATRIX CONSTRUCTION STEP
     */
    auto A = Alien::Matrix(mdist);

    /* FILLING STEP */

    alien_info([&] { cout() << "DIRECT ONE STEP FILLING PHASE";}) ;

    auto tag = Alien::DirectMatrixOptions::eResetValues;
    {
      auto builder = Alien::DirectMatrixBuilder(A, tag, Alien::DirectMatrixOptions::SymmetricFlag::eUnSymmetric);

      // RESERVE 5 non zero entries per row
      builder.reserve(5);
      builder.allocate();

      // LOOP FOLLOWING Y AXE
      for (int j = first_j; j < last_j; ++j) {
        // LOOP FOLLOWING X AXE
        for (int i = 0; i < Nx; ++i) {
          auto n_uid = node_uid(i, j);
          auto n_lid = uid2lid[n_uid];
          auto irow = allUIndex[n_lid];

          // DIAGONAL FILLING
          builder(irow, irow) = 4;

          // OFF DIAG FILLING
          // On bottom
          if (j > 0) {
            auto off_uid = node_uid(i, j - 1);
            auto off_lid = uid2lid[off_uid];
            auto jcol = allUIndex[off_lid];
            if (jcol != -1)
              builder(irow, jcol) = -1;
          }
          // On the left size
          if (i > 0) {
            auto off_uid = node_uid(i - 1, j);
            auto off_lid = uid2lid[off_uid];
            auto jcol = allUIndex[off_lid];
            if (jcol != -1)
              builder(irow, jcol) = -1;
          }
          // on the right side
          if (i < Nx - 1) {
            auto off_uid = node_uid(i + 1, j);
            auto off_lid = uid2lid[off_uid];
            auto jcol = allUIndex[off_lid];
            if (jcol != -1)
              builder(irow, jcol) = -1;
          }
          // On the top
          if (j < Ny - 1) {
            auto off_uid = node_uid(i, j + 1);
            auto off_lid = uid2lid[off_uid];
            auto jcol = allUIndex[off_lid];
            if (jcol != -1)
              builder(irow, jcol) = -1;
          }
        }
      }
    }
    
    
Vector
------

The Vector concept represents the set of unknowns :math:`x=(x_i)` element of a linear space :math:`S_X`.

.. code-block:: bash

    /*
     * VECTOR CONSTRUCTION
     */
    auto B = Alien::Vector(vdist);

    // VECTOR FILLING STEP
    {
      Alien::VectorWriter writer(B);

      // LOOP ALONG Y AXE
      for (int j = first_j; j < last_j; ++j) {
        // LOOP ALONG X AXE
        for (int i = 0; i < Nx; ++i) {
          auto n_uid = node_uid(i, j);
          auto n_lid = uid2lid[n_uid];
          auto irow = allUIndex[n_lid];

          writer[irow] = 1. / (1. + i + j);
        }
      }
    }

    // VECTOR ACCESSOR
    {
      Alien::LocalVectorReader reader(B);
      for (int i = 0; i < reader.size(); ++i) {
        trace_mng->info() << "B[" << i << "]=" << reader[i];
      }
    }


Linear Systems resolution
-------------------------

A linear system is reprensented by a matrix :math:`A`, and two vectors :math:`B` and :math:`X` where :math:`B` is the system right hand side and :math:`X` the solution.

Solving the linear system consists in finding the solution X such that :math:`A*X=B` applying a linear solver algorithm.


.. code-block:: bash

    /*
     * LINEAR SYSTEM CONSTRUCTION
     */
     
    auto A = Alien::Matrix(mdist);
    auto B = Alien::Vector(vdist);
    auto X = Alien::Vector(vdist);
    
    auto solver = createSolver(/*  ... */) ;
    
    solver->init() ;
    
    solver->solve(matrixA, vectorB, vectorX);
      
    Alien::SolverStatus status = solver->getStatus();
    if (status.succeeded) 
    {
        alien_info()([&]{ cout()<<"SOLVER HAS  SUCCEEDED";}) ;

        SimpleCSRLinearAlgebra alg;
        Alien::Vector vectorR(m_vdist);
        alg.mult(matrixA, vectorX, vectorR);
        alg.axpy(-1., vectorB, vectorR);
        Real res = alg.norm2(vectorR);
        alien_info([&] cout() << "RES : " << res;}) ;
      }
      else
        alien_info()([&]{ cout()<<"SOLVER FAILED";}) ;
      solver->getSolverStat().print(Universe().traceMng(), status, "Linear Solver : ");
    }
    
    solver->end();
    