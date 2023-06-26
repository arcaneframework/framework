/*
 * TrilinosInternalSolver.h
 *
 *  Created on: Dec 16, 2019
 *      Author: gratienj
 */

#ifndef MODULES_TRILINOS_SRC_ALIEN_KERNELS_TRILINOS_LINEARSOLVER_TRILINOSINTERNALSOLVER_H_
#define MODULES_TRILINOS_SRC_ALIEN_KERNELS_TRILINOS_LINEARSOLVER_TRILINOSINTERNALSOLVER_H_

#ifdef ALIEN_USE_TRILINOS
#include <Kokkos_DefaultNode.hpp>

#include <Tpetra_Version.hpp>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>
#define HAVE_MUELU
#ifdef HAVE_MUELU 
#include <MueLu.hpp>

#include <MueLu_BaseClass.hpp>
#ifdef HAVE_MUELU_EXPLICIT_INSTANTIATION
#include <MueLu_ExplicitInstantiation.hpp>
#endif
#include <MueLu_Level.hpp>
#include <MueLu_MutuallyExclusiveTime.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_Utilities.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#endif

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <MatrixMarket_Tpetra.hpp>

#if defined(HAVE_MUELU_AMGX)
//#include <MueLu_AMGXOperator.hpp>
#include <alien/kernels/trilinos/linear_solver/ALIEN_AMGXOperator_decl.hpp>
#include <alien/kernels/trilinos/linear_solver/ALIEN_AMGXOperator_def.hpp>
#endif
#endif

class IOptionsTrilinosSolver;

BEGIN_TRILINOSINTERNAL_NAMESPACE

template <typename TagT> class SolverInternal
{
 public:
  typedef typename TrilinosMatrix<Real, TagT>::MatrixInternal MatrixInternalType;

  typedef typename MatrixInternalType::matrix_type matrix_type;
  typedef typename matrix_type::scalar_type scalar_type;
  typedef typename matrix_type::local_ordinal_type local_ordinal_type;
  typedef typename matrix_type::global_ordinal_type global_ordinal_type;

  typedef typename matrix_type::node_type node_type;
  typedef Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>
      vec_type;

  typedef typename TrilinosVector<Real, TagT>::VectorInternal VectorInternalType;
  typedef typename VectorInternalType::vector_type vector_type;

  typedef Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>
      op_type;
  typedef Ifpack2::Preconditioner<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>
      prec_type;
#ifdef HAVE_MUELU_AMGX
  typedef MueLu::ALIEN_AMGXOperator<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>
      amgx_prec_type;
#endif
  typedef Belos::SolverManager<scalar_type, vec_type, op_type> solver_type;

  typedef Teuchos::ScalarTraits<scalar_type> STS;
  typedef typename STS::coordinateType real_type;
  typedef typename STS::magnitudeType magnitude_type;
  typedef Teuchos::ScalarTraits<magnitude_type> STM;

  typedef Tpetra::MultiVector<real_type, 
                              local_ordinal_type, 
                              global_ordinal_type,
                              node_type> RealValuedMultiVector;
  
  typedef Teuchos::RCP<RealValuedMultiVector> coord_type ;

  struct Status
  {
    bool m_converged = false;
    int m_num_iters = 0;
    magnitude_type m_residual = -1.;
  };

 private:
  std::string m_precond_name;
  Teuchos::RCP<Teuchos::ParameterList> m_precond_parameters;

  std::string m_solver_name;
  Teuchos::RCP<Teuchos::ParameterList> m_solver_parameters;

  bool m_use_amgx = false;
#ifdef HAVE_MUELU_AMGX
  amgx_prec_type* m_amgx_M_ptr = nullptr;
  Teuchos::RCP<op_type> m_amgx_M;
#endif
  Teuchos::RCP<const Teuchos::Comm<int>> m_comm;

 public:
  void initPrecondParameters(IOptionsTrilinosSolver* options, MPI_Comm const* comm)
  {
    using Teuchos::ParameterList;
    using Teuchos::parameterList;
    using Teuchos::RCP;
    using Teuchos::rcp;

    m_precond_name = TrilinosOptionTypes::precondName(options->preconditioner());
    Teuchos::ParameterList paramList;

    m_precond_parameters = rcp(new ParameterList("Preconditioner"));
    switch (options->preconditioner()) {
    case TrilinosOptionTypes::None:
      break;
    case TrilinosOptionTypes::Relaxation:
      // m_precond_parameters->set ("Ifpack2::Preconditioner", m_precond_name);
      if (options->relaxation().size() > 0) {
        m_precond_parameters->set(
            "relaxation: type", options->relaxation()[0]->type().localstr());
        m_precond_parameters->set(
            "relaxation: sweeps", options->relaxation()[0]->sweeps());
        m_precond_parameters->set(
            "relaxation: damping factor", options->relaxation()[0]->dampingFactor());
        m_precond_parameters->set(
            "relaxation: backward mode", options->relaxation()[0]->backwardMode());
        m_precond_parameters->set(
            "relaxation: use l1", options->relaxation()[0]->useL1());
        m_precond_parameters->set(
            "relaxation: l1 eta", options->relaxation()[0]->l1Eta());
        m_precond_parameters->set("relaxation: zero starting solution",
            options->relaxation()[0]->zeroStartingSolution());
      }
      break;
    case TrilinosOptionTypes::Chebyshev:
      // m_precond_parameters->set ("Ifpack2::Preconditioner", m_precond_name);
      // m_precond_parameters->set( "debug",true) ;
      // m_precond_parameters->set( "chebyshev: textbook algorithm",true) ;
      if (options->chebyshev().size() > 0) {
        m_precond_parameters->set("chebyshev: degree", options->chebyshev()[0]->degree());
        m_precond_parameters->set(
            "chebyshev: min eigenvalue", options->chebyshev()[0]->minEigenvalue());
        m_precond_parameters->set(
            "chebyshev: max eigenvalue", options->chebyshev()[0]->maxEigenvalue());
        m_precond_parameters->set(
            "chebyshev: ratio eigenvalue", options->chebyshev()[0]->ratioEigenvalue());
        m_precond_parameters->set("chebyshev: eigenvalue max iterations",
            options->chebyshev()[0]->eigenvalueMaxIterations());
        m_precond_parameters->set(
            "chebyshev: boost factor", options->chebyshev()[0]->boostFactor());
        m_precond_parameters->set("chebyshev: zero starting solution",
            options->chebyshev()[0]->zeroStartingSolution());
      }
      break;
    case TrilinosOptionTypes::ILUK:
      m_precond_parameters->set("Ifpack2::Preconditioner", m_precond_name);
      if (options->iluk().size() > 0) {
        m_precond_parameters->set(
            "fact: iluk level-of-fill", options->iluk()[0]->levelOfFill());
        m_precond_parameters->set("fact: relax value", options->iluk()[0]->relaxValue());
        m_precond_parameters->set(
            "fact: absolute threshold", options->iluk()[0]->absoluteThreshold());
        m_precond_parameters->set(
            "fact: relative threshold", options->iluk()[0]->relativeThreshold());
      }
      break;
    case TrilinosOptionTypes::ILUT:
      m_precond_parameters->set("Ifpack2::Preconditioner", m_precond_name);
      if (options->ilut().size() > 0) {
        m_precond_parameters->set(
            "fact: ilut level-of-fill", options->ilut()[0]->levelOfFill());
        m_precond_parameters->set(
            "fact: drop tolerance", options->ilut()[0]->dropTolerance());
        m_precond_parameters->set(
            "fact: absolute threshold", options->ilut()[0]->absoluteThreshold());
        m_precond_parameters->set(
            "fact: relative threshold", options->ilut()[0]->relativeThreshold());
        m_precond_parameters->set("fact: relax value", options->ilut()[0]->relaxValue());
      }
      break;
    case TrilinosOptionTypes::FILU:
      m_precond_parameters->set("Ifpack2::Preconditioner", m_precond_name);
      if (options->filu().size() > 0) {
        m_precond_parameters->set("level", options->filu()[0]->level());
        m_precond_parameters->set("damping factor", options->filu()[0]->dampingFactor());
        m_precond_parameters->set(
            "triangular solve iterations", options->filu()[0]->solverNbIterations());
        m_precond_parameters->set("sweeps", options->filu()[0]->factorNbIterations());
      }
      break;
    case TrilinosOptionTypes::Schwarz:
      // m_precond_parameters->set ("Ifpack2::Preconditioner", m_precond_name);
      if (options->schwarz().size() > 0) {
        m_precond_parameters->set("schwarz: subdomain solver name",
            options->schwarz()[0]->subdomainSolver().localstr());
        m_precond_parameters->set(
            "schwarz: combine mode", options->schwarz()[0]->combineMode().localstr());
        m_precond_parameters->set(
            "schwarz: num iterations", options->schwarz()[0]->numIterations());
        if (options->schwarz()[0]->subdomainSolver() == "RELAXATION") {
          if (options->relaxation().size() > 0) {
            ParameterList plist;
            // plist.set ("Ifpack2::Preconditioner", "ILUT");
            plist.set("relaxation: type", options->relaxation()[0]->type().localstr());
            plist.set("relaxation: sweeps", options->relaxation()[0]->sweeps());
            plist.set(
                "relaxation: damping factor", options->relaxation()[0]->dampingFactor());
            plist.set(
                "relaxation: backward mode", options->relaxation()[0]->backwardMode());
            plist.set("relaxation: use l1", options->relaxation()[0]->useL1());
            plist.set("relaxation: l1 eta", options->relaxation()[0]->l1Eta());

            m_precond_parameters->set("schwarz: subdomain solver parameters", plist);
          }
        }
        if (options->schwarz()[0]->subdomainSolver() == "ILUT") {
          if (options->ilut().size() > 0) {
            ParameterList plist;
            // plist.set ("Ifpack2::Preconditioner", "ILUT");
            plist.set("fact: ilut level-of-fill", options->ilut()[0]->levelOfFill());
            plist.set("fact: drop tolerance", options->ilut()[0]->dropTolerance());
            plist.set(
                "fact: absolute threshold", options->ilut()[0]->absoluteThreshold());
            plist.set(
                "fact: relative threshold", options->ilut()[0]->relativeThreshold());
            plist.set("fact: relax value", options->ilut()[0]->relaxValue());
            m_precond_parameters->set("schwarz: subdomain solver parameters", plist);
          }
        }
        if (options->schwarz()[0]->subdomainSolver() == "ILUK") {
          if (options->iluk().size() > 0) {
            ParameterList plist;
            // plist.set ("Ifpack2::Preconditioner", "ILUK");
            plist.set("fact: iluk level-of-fill", options->iluk()[0]->levelOfFill());
            plist.set("fact: relax value", options->iluk()[0]->relaxValue());
            plist.set(
                "fact: absolute threshold", options->iluk()[0]->absoluteThreshold());
            plist.set(
                "fact: relative threshold", options->iluk()[0]->relativeThreshold());
            m_precond_parameters->set("schwarz: subdomain solver parameters", plist);
          }
        }
        if(options->schwarz()[0]->subdomainSolver()==TrilinosOptionTypes::precondName(TrilinosOptionTypes::FILU))
        {
            if(options->filu().size()>0)
            {
              ParameterList plist ;
              //plist.set ("Ifpack2::Preconditioner", "FILU");
              plist.set("level",options->filu()[0]->level()) ;
              plist.set("damping factor",options->filu()[0]->dampingFactor()) ;
              plist.set("triangular solve iterations",options->filu()[0]->solverNbIterations()) ;
              plist.set("sweeps",options->filu()[0]->factorNbIterations()) ;
              m_precond_parameters->set("schwarz: subdomain solver parameters",plist);
            }
        }
        if (options->schwarz()[0]->subdomainSolver() == "CHEBYSHEV") {
          if (options->chebyshev().size() > 0) {
            ParameterList plist;
            // plist.set ("Ifpack2::Preconditioner", "CHEBYSHEV");
            plist.set("chebyshev: degree", options->chebyshev()[0]->degree());
            plist.set(
                "chebyshev: min eigenvalue", options->chebyshev()[0]->minEigenvalue());
            plist.set(
                "chebyshev: max eigenvalue", options->chebyshev()[0]->maxEigenvalue());
            plist.set("chebyshev: ratio eigenvalue",
                options->chebyshev()[0]->ratioEigenvalue());
            plist.set("chebyshev: eigenvalue max iterations",
                options->chebyshev()[0]->eigenvalueMaxIterations());
            plist.set("chebyshev: boost factor", options->chebyshev()[0]->boostFactor());
            plist.set("chebyshev: zero starting solution",
                options->chebyshev()[0]->zeroStartingSolution());
            m_precond_parameters->set("schwarz: subdomain solver parameters", plist);
          }
        }
      }
      break;
    case TrilinosOptionTypes::MLPC:
    case TrilinosOptionTypes::MueLuPC:
      if (options->muelu().size() > 0) {
        m_precond_parameters->set(
            "verbosity", options->muelu()[0]->verbosity().localstr());
        if (options->muelu()[0]->amgx().size() > 0) {
          if (options->muelu()[0]->amgx()[0]->enable()) {
            m_use_amgx = true;
            m_precond_parameters->set("use external multigrid package", "amgx");
            ParameterList plist;
            if (options->muelu()[0]->amgx()[0]->config().size() > 0) {
              plist.set("config_version", 2);
              for (auto const& param :
                  options->muelu()[0]->amgx()[0]->config()[0]->parameter()) {
                plist.set(param->key().localstr(), param->value().localstr());
              }
              /*
              config_version=2,determinism_flag=1,solver(amg)=AMG,amg:algorithm=AGGREGATION,amg:selector=SIZE_2,
              amg:max_iters=1,amg:cycle=V,amg:presweeps=2,amg:postsweeps=2,amg:relaxation_factor=0.75,
              amg:coarsest_sweeps=2,amg:smoother=BLOCK_JACOBI
              */
              /*config_version=2,solver(amg)=AMG,amg:selector=PMIS,amg:max_iters=1,amg:max_levels=24,
                amg:cycle=V,amg:presweeps=2,amg:postsweeps=2,amg:coarsest_sweeps=2,
                amg:interpolator=D2,amg:min_coarse_rows=2,amg:interp_max_elements=4,amg:coarse_solver=NOSOLVER,
                amg:smoother=BLOCK_JACOBI,amg:structure_reuse_levels=-1
              */
            } else
              plist.set("json file",
                  options->muelu()[0]->amgx()[0]->parameterFile().localstr());
            m_precond_parameters->set("amgx:params", plist);
          }
        } else {
          m_precond_parameters->set("max levels", options->muelu()[0]->maxLevel());
          m_precond_parameters->set(
              "cycle type", options->muelu()[0]->cycleType().localstr());
          m_precond_parameters->set(
              "problem: symmetric", options->muelu()[0]->symmetric());
          if (options->muelu()[0]->repartition().size() > 0) {
            m_precond_parameters->set(
                "repartition: enable", options->muelu()[0]->repartition()[0]->enable());
            m_precond_parameters->set("repartition: start level",
                options->muelu()[0]->repartition()[0]->startLevel());
            m_precond_parameters->set("repartition: min rows per proc",
                options->muelu()[0]->repartition()[0]->minRowsPerProc());
            m_precond_parameters->set("repartition: max imbalance", 1.1);
            m_precond_parameters->set("repartition: partitioner", "zoltan2");
            m_precond_parameters->set("repartition: rebalance P and R", false);
            m_precond_parameters->set("repartition: remap parts", false);

            ParameterList plist;
            // plist.set("algorithm","multijagged") ;
            // plist.set("algorithm","rcb") ;
            // plist.set("algorithm","scotch") ;
            plist.set("algorithm", "parmetis");
            m_precond_parameters->set("repartition: params", plist);
          }
          if (options->muelu()[0]->xmlParameterFile().size() > 0)
            m_precond_parameters->set("xml parameter file",
                options->muelu()[0]->xmlParameterFile()[0].localstr());
          m_precond_parameters->set(
              "smoother: type", options->muelu()[0]->smootherType().localstr());
          if (options->muelu()[0]->smootherType() == "RELAXATION") {
            if (options->relaxation().size() > 0) {
              ParameterList plist;
              // plist.set ("Ifpack2::Preconditioner", "ILUT");
              plist.set("relaxation: type", options->relaxation()[0]->type().localstr());
              plist.set("relaxation: sweeps", options->relaxation()[0]->sweeps());
              plist.set("relaxation: damping factor",
                  options->relaxation()[0]->dampingFactor());
              plist.set(
                  "relaxation: backward mode", options->relaxation()[0]->backwardMode());
              plist.set("relaxation: use l1", options->relaxation()[0]->useL1());
              plist.set("relaxation: l1 eta", options->relaxation()[0]->l1Eta());

              m_precond_parameters->set("smoother: params", plist);
            }
          }
          if (options->muelu()[0]->smootherType() == "ILUT") {
            if (options->ilut().size() > 0) {
              ParameterList plist;
              // plist.set ("Ifpack2::Preconditioner", "ILUT");
              plist.set("fact: ilut level-of-fill", options->ilut()[0]->levelOfFill());
              plist.set("fact: drop tolerance", options->ilut()[0]->dropTolerance());
              plist.set(
                  "fact: absolute threshold", options->ilut()[0]->absoluteThreshold());
              plist.set(
                  "fact: relative threshold", options->ilut()[0]->relativeThreshold());
              plist.set("fact: relax value", options->ilut()[0]->relaxValue());
              m_precond_parameters->set("smoother: params", plist);
            }
          }
          if (options->muelu()[0]->smootherType() == "ILUK") {
            if (options->iluk().size() > 0) {
              ParameterList plist;
              // plist.set ("Ifpack2::Preconditioner", "ILUK");
              plist.set("fact: iluk level-of-fill", options->iluk()[0]->levelOfFill());
              plist.set("fact: relax value", options->iluk()[0]->relaxValue());
              plist.set(
                  "fact: absolute threshold", options->iluk()[0]->absoluteThreshold());
              plist.set(
                  "fact: relative threshold", options->iluk()[0]->relativeThreshold());
              m_precond_parameters->set("smoother: params", plist);
            }
          }
          if (options->muelu()[0]->smootherType() == "CHEBYSHEV") {
            if (options->chebyshev().size() > 0) {
              ParameterList plist;
              // plist.set ("Ifpack2::Preconditioner", "CHEBYSHEV");
              plist.set("chebyshev: degree", options->chebyshev()[0]->degree());
              plist.set(
                  "chebyshev: min eigenvalue", options->chebyshev()[0]->minEigenvalue());
              plist.set(
                  "chebyshev: max eigenvalue", options->chebyshev()[0]->maxEigenvalue());
              plist.set("chebyshev: ratio eigenvalue",
                  options->chebyshev()[0]->ratioEigenvalue());
              plist.set("chebyshev: eigenvalue max iterations",
                  options->chebyshev()[0]->eigenvalueMaxIterations());
              plist.set(
                  "chebyshev: boost factor", options->chebyshev()[0]->boostFactor());
              plist.set("chebyshev: zero starting solution",
                  options->chebyshev()[0]->zeroStartingSolution());
              m_precond_parameters->set("smoother: params", plist);
            }
          }
          m_precond_parameters->set(
              "smoother: overlap", options->muelu()[0]->smootherOverlap());

          m_precond_parameters->set(
              "coarse: max size", options->muelu()[0]->maxCoarseSize());
          m_precond_parameters->set(
              "coarse: type", options->muelu()[0]->coarseType().localstr());
          m_precond_parameters->set(
              "coarse: overlap", options->muelu()[0]->coarseOverlap());

          m_precond_parameters->set(
              "aggregation: type", options->muelu()[0]->aggregationType().localstr());
          // m_precond_parameters->set("aggregation: mode",
          // options->muelu()[0]->aggregationMode().localstr());
          m_precond_parameters->set("aggregation: ordering",
              options->muelu()[0]->aggregationOrdering().localstr());
          m_precond_parameters->set("aggregation: drop scheme",
              options->muelu()[0]->aggregationDropScheme().localstr());
          m_precond_parameters->set(
              "aggregation: drop tol", options->muelu()[0]->aggregationDropTol());
          m_precond_parameters->set(
              "aggregation: min agg size", options->muelu()[0]->aggregationMinAggSize());
          m_precond_parameters->set(
              "aggregation: max agg size", options->muelu()[0]->aggregationMaxAggSize());
          m_precond_parameters->set("aggregation: Dirichlet threshold",
              options->muelu()[0]->aggregationDirichletThreshold());

          m_precond_parameters->set("multigrid algorithm",
              options->muelu()[0]->multigridAlgorithm().localstr());
          m_precond_parameters->set(
              "sa: damping factor", options->muelu()[0]->saDamplingFactor());
        }
      }
      break;
    default:
      break;
    }
#ifdef HAVE_MUELU_AMGX
    if (m_use_amgx) {
      using Teuchos::RCP;
      using Teuchos::rcp;
      using Teuchos::Comm;
      using Teuchos::MpiComm;
      std::cout<<"INIT AMGX OPERATOR : "<<comm<<std::endl ;
      m_comm = rcp(new MpiComm<int>(*comm));
      m_amgx_M_ptr = new amgx_prec_type(m_comm, *m_precond_parameters) ;
      m_amgx_M = rcp(m_amgx_M_ptr);
    }
#endif
  }

  void initSolverParameters(IOptionsTrilinosSolver* options)
  {
    using Teuchos::ParameterList;
    using Teuchos::parameterList;
    using Teuchos::RCP;
    using Teuchos::rcp;
    m_solver_name = TrilinosOptionTypes::solverName(options->solver());

    // Make an empty new parameter list.
    m_solver_parameters = rcp(new ParameterList());

    int level = Belos::Errors + Belos::Warnings;
    if (options->output() > 1)
      level += Belos::FinalSummary + Belos::StatusTestDetails;
    if (options->output() > 2)
      level += Belos::TimingDetails;
    if (options->output() > 3)
      level += Belos::Debug;

    m_solver_parameters->set("Verbosity", level);
    m_solver_parameters->set("Output Frequency", 1);
    m_solver_parameters->set("Output Style", Belos::General);
    m_solver_parameters->set("Num Blocks", options->maxRestartIterationNum());
    m_solver_parameters->set("Maximum Iterations", options->maxIterationNum());
    m_solver_parameters->set("Convergence Tolerance", options->stopCriteriaValue());
  }

  Teuchos::RCP<Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>>
  createPreconditioner(matrix_type& A, 
                       coord_type& A_coordinates, 
                       const std::string& precondType,
                       Teuchos::RCP<Teuchos::ParameterList> plist,
                       std::ostream& out,
                       std::ostream& err)
  {
    using Teuchos::ParameterList;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::Time;
    using Teuchos::TimeMonitor;
    using std::endl;

    // Fetch the typedefs defined by Tpetra::CrsMatrix.

    // Ifpack2's generic Preconditioner interface implements
    // Tpetra::Operator.  A Tpetra::Operator is an abstraction of a
    // function mapping a (Multi)Vector to a (Multi)Vector, with the
    // option of applying the transpose or conjugate transpose of the
    // operator.  Tpetra::CrsMatrix implements Operator as well.
    typedef Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type,
        node_type>
        op_type;

    // These are just some convenience typedefs.

    // An Ifpack2::Preconditioner is-a Tpetra::Operator.  Ifpack2
    // creates a Preconditioner object, but users of iterative methods
    // want a Tpetra::Operator.  That's why create() returns an Operator
    // instead of a Preconditioner.

    // Create timers to show how long it takes for Ifpack2 to do various operations.
    RCP<Time> initTimer =
        TimeMonitor::getNewCounter("Ifpack2::Preconditioner::initialize");
    RCP<Time> computeTimer =
        TimeMonitor::getNewCounter("Ifpack2::Preconditioner::compute");
    RCP<Time> condestTimer =
        TimeMonitor::getNewCounter("Ifpack2::Preconditioner::condest");

    out << "Creating preconditioner : " << precondType << endl
        << "-- Configuring" << endl;
    //
    // Create the preconditioner and set parameters.
    //
    // This doesn't actually _compute_ the preconditioner.
    // It just sets up the specific type of preconditioner and
    // its associated parameters (which depend on the type).
    //
    // RCP<muelu_prec_type> mueLuPreconditioner ;
    if (precondType.compare("MueLu") == 0) {
#ifdef HAVE_MUELU
      typedef MueLu::TpetraOperator<scalar_type, local_ordinal_type, global_ordinal_type,
          node_type>
          prec_type;
      RCP<op_type> opMat(rcpFromRef(A));
      //RCP<prec_type> prec = MueLu::CreateTpetraPreconditioner(opMat, *plist,A_coordinates);
      Teuchos::ParameterList& userList = plist->sublist("user data");
      if (Teuchos::nonnull(A_coordinates)) {
        userList.set<RCP<Tpetra::MultiVector<typename Teuchos::ScalarTraits<scalar_type>::coordinateType,local_ordinal_type,global_ordinal_type,node_type> > >("Coordinates", A_coordinates);
      }
      RCP<prec_type> prec = MueLu::CreateTpetraPreconditioner(opMat, *plist);
      return prec;
#else
      return RCP<op_type>() ;
#endif
    } else {
      typedef Ifpack2::Preconditioner<scalar_type, local_ordinal_type,
          global_ordinal_type, node_type>
          prec_type;

      RCP<prec_type> prec;
      Ifpack2::Factory factory;
      // Set up the preconditioner of the given type.
      const matrix_type& cA = A;
      prec = factory.create(precondType, rcpFromRef(cA));
      prec->setParameters(*plist);

      out << "-- Initializing" << endl;
      {
        TimeMonitor mon(*initTimer);
        prec->initialize();
      }
      out << "-- Computing" << endl;
      {
        TimeMonitor mon(*computeTimer);
        prec->compute();
      }
      return prec;
    }
  }

  Status solve(std::ostream& out, vec_type& X, const vec_type& B, const op_type& A,
      Teuchos::RCP<op_type> M)
  {
    using Teuchos::ParameterList;
    using Teuchos::parameterList;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcpFromRef; // Make a "weak" RCP from a reference.

    Belos::SolverFactory<scalar_type, vec_type, op_type> factory;
    RCP<Belos::SolverManager<scalar_type, vec_type, op_type>> solver =
        factory.create(m_solver_name, m_solver_parameters);

    // Create a LinearProblem struct with the problem to solve.
    // A, X, B, and M are passed by (smart) pointer, not copied.
    typedef Belos::LinearProblem<scalar_type, vec_type, op_type> problem_type;
    RCP<problem_type> problem =
        rcp(new problem_type(rcpFromRef(A), rcpFromRef(X), rcpFromRef(B)));

    if (m_precond_name.compare("MueLU") == 0)
      problem->setLeftPrec(M);
    else
      problem->setRightPrec(M);

    // Tell the LinearProblem to make itself ready to solve.
    problem->setProblem();

    // Tell the solver what problem you want to solve.
    solver->setProblem(problem);

    Belos::ReturnType result = solver->solve();

    Status status;
    status.m_converged = result == Belos::Converged;
    status.m_num_iters = solver->getNumIters();
    status.m_residual = solver->achievedTol();
    return status;
  }

  Status solve(matrix_type& A, 
               coord_type coordinates,
               const vector_type& B, 
               vector_type& X)
  {
    using Teuchos::parameterList;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcpFromRef; // Make a "weak" RCP from a reference.
    using Teuchos::ParameterList;
    using Teuchos::parameterList;

    // Compute the preconditioner using the matrix A.
    // The matrix A itself is not modified.
    RCP<op_type> M;
#ifdef HAVE_MUELU_AMGX
    if (m_use_amgx) {
      m_amgx_M_ptr->init(rcpFromRef(A));
      M = m_amgx_M;
    } else
#endif
      M = createPreconditioner(A,coordinates, m_precond_name, m_precond_parameters, std::cout, std::cerr);

    // Create a LinearProblem struct with the problem to solve.
    // A, X, B, and M are passed by (smart) pointer, not copied.
    typedef Belos::LinearProblem<scalar_type, vec_type, op_type> problem_type;
    RCP<problem_type> problem =
        rcp(new problem_type(rcpFromRef(A), rcpFromRef(X), rcpFromRef(B)));

    if (m_precond_name.compare("MueLU") == 0)
      problem->setLeftPrec(M);
    else
      problem->setRightPrec(M);

    // Tell the LinearProblem to make itself ready to solve.
    problem->setProblem();

    // the list of solver parameters created above.
    Belos::SolverFactory<scalar_type, vec_type, op_type> factory;
    RCP<Belos::SolverManager<scalar_type, vec_type, op_type>> solver =
        factory.create(m_solver_name, m_solver_parameters);

    // Tell the solver what problem you want to solve.
    solver->setProblem(problem);

    // Tpetra::MatrixMarket::Writer<matrix_type>::writeSparseFile("A.txt",A);
    // Tpetra::MatrixMarket::Writer<matrix_type>::writeDenseFile("B.txt",B);
    // Tpetra::MatrixMarket::Writer<matrix_type>::writeDenseFile("X.txt",X);

    Belos::ReturnType result = solver->solve();

    Status status;
    status.m_converged = result == Belos::Converged;
    status.m_num_iters = solver->getNumIters();
    status.m_residual = solver->achievedTol();
    return status;
  }
};
END_TRILINOSINTERNAL_NAMESPACE

#endif /* MODULES_TRILINOS_SRC_ALIEN_KERNELS_TRILINOS_LINEARSOLVER_TRILINOSINTERNALSOLVER_H_ \
          */
