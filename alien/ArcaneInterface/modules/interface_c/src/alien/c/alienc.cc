// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* alienc                                         (C) 2000-2024              */
/*                                                                           */
/* Interface C for alien                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include <mpi.h>
#include <assert.h>
#include <memory>
#include <vector>
#include <set>
#include <string>

#include <unordered_map>

#include <arccore/base/StringBuilder.h>
#include <alien/AlienLegacyConfig.h>
#include <alien/data/Universe.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/index_manager/functional/DefaultAbstractFamily.h>
#include <alien/index_manager/functional/AbstractItemFamily.h>
#include <alien/index_manager/functional/DefaultIndexManager.h>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/trace/ITraceMng.h>

#include <alien/AlienInterfaceCExport.h>

#include <alien/ref/AlienRefSemantic.h>
//#include <alien/AlienExternalPackages.h>



#include <alien/expression/solver/ILinearSolver.h>
#include <alien/utils/parameter_manager/BaseParameterManager.h>
#include <alien/core/backend/SolverFactory.h>

#if defined ALIEN_USE_MTL4 || defined ALIEN_USE_PETSC || defined ALIEN_USE_HYPRE
#include <alien/AlienExternalPackages.h>
#endif
#if defined ALIEN_USE_IFPSOLVER || defined ALIEN_USE_MCGSOLVER
#include <alien/AlienIFPENSolvers.h>
#endif

#include <boost/program_options/value_semantic.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/program_options/parsers.hpp>


class AlienManager
{
public :
  typedef int       id_type ;
  typedef long int uid_type ;

  static AlienManager* m_singleton ;

  static void initialize() {
    if(m_singleton)
      delete m_singleton ;
    m_singleton = new AlienManager() ;
  }

  static void finalize() {
    if(m_singleton)
      delete m_singleton ;
    m_singleton = nullptr ;
  }


  static AlienManager* instance() {
    return m_singleton ;
  }


  AlienManager() ;

  Arccore::MessagePassing::IMessagePassingMng*
  parallelMng()
  {
    return m_parallel_mng;
  }

  Arccore::ITraceMng*
  traceMng()
  {
    return m_trace_mng;
  }

  class LinearSystem
  {
    public :
    LinearSystem() {}

    LinearSystem(MPI_Comm comm)
    : m_parallel(true)
    , m_comm(comm)
    {}

    void init(int global_size, int local_size, uid_type* row_uids, int nb_ghosts,
        uid_type* ghost_uids,
        int* ghost_owners) ;


    void defineMatrixProfile(int local_nrows,
                             uid_type* row_uids,
                             int* row_offset,
                             uid_type* col_uids) ;


    void setMatrixValues(int local_nrows,
                         uid_type* row_uids,
                         int* row_offset,
                         uid_type* col_uids,
                         double* values) ;


    void setRHSValues(int local_nrows,
                      uid_type* row_uids,
                      double const* values) ;

    void getSolutionValues(int local_nrows,
                           uid_type* row_uids,
                           double* values) ;

    Alien::Matrix const& getA() const {
      return *m_A;
    }

    Alien::Vector& getX() {
      return *m_x;
    }
    Alien::Vector const& getB() const {
      return *m_b;
    }

    private :

    bool     m_parallel = false ;
    MPI_Comm m_comm ;

    std::unique_ptr<Alien::Space>              m_space;
    std::unique_ptr<Alien::AbstractItemFamily> m_item_family ;
    std::unique_ptr<Alien::IndexManager>       m_index_manager;
    Alien::ScalarIndexSet                      m_index_set ;
    Arccore::UniqueArray<Arccore::Integer>     m_allUIndex ;
    std::map<uid_type,int>                     m_uid2lid ;
    std::unique_ptr<Alien::MatrixDistribution> m_matrix_dist;
    std::unique_ptr<Alien::VectorDistribution> m_vector_dist;
    std::unique_ptr<Alien::Matrix>             m_A;
    std::unique_ptr<Alien::Vector>             m_x;
    std::unique_ptr<Alien::Vector>             m_b;
  };

  int createNewLinearSystem(MPI_Comm comm)
  {
    int id = m_linear_systems.size() ;
    m_linear_systems.push_back(std::make_unique<LinearSystem>(comm)) ;

    return id ;
  }

  int destroyLinearSystem(int system_id)
  {
    assert((std::size_t)system_id < m_linear_systems.size()) ;
    m_linear_systems[system_id].reset() ;
    return 0 ;
  }

  LinearSystem* getLinearSystem(int system_id)
  {
    assert((std::size_t)system_id < m_linear_systems.size()) ;
    return m_linear_systems[system_id].get() ;
  }

  class ParamSystem
  {
  public :
    template<typename T>
    void addToCommandLine(std::string const& key, T value)
    {
      {
        std::stringstream token;
        token<<"--"<<key;
        m_command_line.push_back(token.str());
      }
      {
        std::stringstream token;
        token<<value;
        m_command_line.push_back(token.str());
      }
    }

    void setParam(std::string const& key,std::string const& value)
    {
      m_string_params[key] = value ;
      addToCommandLine(key,value) ;
    }

    void setParam(std::string const& key,int value)
    {
      m_integer_params[key] = value ;
      addToCommandLine(key,value) ;
    }

    void setParam(std::string const& key,double value)
    {
      m_double_params[key] = value ;
      addToCommandLine(key,value) ;
    }

    std::vector<const char *> commandLine() const {
      std::vector<const char *> command_line(m_command_line.size()) ;
      for(std::size_t i=0;i<m_command_line.size();++i)
      {
        command_line[i] = m_command_line[i].c_str() ;
      }
      return command_line ;
    }

  public:
    std::map<std::string,std::string> m_string_params ;
    std::map<std::string,int>         m_integer_params ;
    std::map<std::string,double>      m_double_params ;
    std::vector<std::string>          m_command_line = {"ALIENCommndLine"} ;
  };


  int createNewParamSystem()
  {
    int id = m_param_systems.size() ;
    m_param_systems.push_back(std::make_unique<ParamSystem>()) ;

    return id ;
  }

  int destroyParamSystem(int system_id)
  {
    assert((std::size_t)system_id < m_param_systems.size()) ;
    m_param_systems[system_id].reset() ;
    return 0 ;
  }

  ParamSystem* getParamSystem(int system_id)
  {
    assert((std::size_t)system_id < m_param_systems.size()) ;
    return m_param_systems[system_id].get() ;
  }


  class LinearSolver
  {
    public :

    LinearSolver(MPI_Comm comm, const char* config_file)
    : m_parallel(true)
    , m_comm(comm)
    , m_config_file(config_file)
    {}

    void init(int argc, const char** argv);

    void init(std::string const& configfile);

    void init(ParamSystem const& param_system);

    int solve(Alien::Matrix const& A, Alien::Vector const& B, Alien::Vector& X);

    void getStatus(int* code, double* residual, int* num_iterations) ;


    private :
    bool     m_parallel      = false ;
    MPI_Comm m_comm ;
    const char* m_config_file = nullptr ;
    std::unique_ptr<Alien::ILinearSolver> m_linear_solver ;
  } ;


  int createNewLinearSolver(MPI_Comm comm, const char* config_file)
  {

    int id = m_linear_solvers.size() ;
    m_linear_solvers.push_back(std::make_unique<LinearSolver>(comm,config_file)) ;
    return id ;
  }

  int destroyLinearSolver(int solver_id)
  {
    assert((std::size_t)solver_id < m_linear_solvers.size()) ;
    m_linear_solvers[solver_id].reset() ;
    return 0 ;
  }

  LinearSolver* getLinearSolver(int solver_id)
  {
    assert((std::size_t)solver_id < m_linear_solvers.size()) ;
    return m_linear_solvers[solver_id].get() ;
  }

private :

  std::vector<std::unique_ptr<LinearSystem> > m_linear_systems ;

  std::vector<std::unique_ptr<ParamSystem> >  m_param_systems ;

  std::vector<std::unique_ptr<LinearSolver> > m_linear_solvers ;


  Arccore::ITraceMng*                              m_trace_mng    = nullptr;
  Arccore::ReferenceCounter<Arccore::ITraceStream> m_ofile ;
  Arccore::MessagePassing::IMessagePassingMng*     m_parallel_mng = nullptr;
};


AlienManager::AlienManager()
{
  m_parallel_mng = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(
        MPI_COMM_WORLD);

  // Gestionnaire de trace
  m_trace_mng = Arccore::arccoreCreateDefaultTraceMng();
  Arccore::StringBuilder filename("alien.log") ;
  if(m_parallel_mng->commSize()>1)
  {
    filename += m_parallel_mng->commRank() ;
    m_ofile = Arccore::ITraceStream::createFileStream(filename.toString()) ;
    m_trace_mng->setRedirectStream(m_ofile.get());
  }
  m_trace_mng->finishInitialize() ;
  m_trace_mng->info()<<"INFO INIT ALIEN MANAGER";
}

void AlienManager::LinearSystem::
init([[maybe_unused]] int global_nrows,
     int local_nrows,
     uid_type* row_uids,
     int nb_ghosts,
     uid_type* ghost_uids,
     int* ghost_owners)
{
  auto* mng = AlienManager::instance() ;
  auto* pm = mng->parallelMng() ;
  int my_rank = pm->commRank() ;
  //auto* tm = mng->traceMng() ;

  m_index_manager.reset(new Alien::IndexManager(mng->parallelMng(),mng->traceMng())) ;
  m_index_manager->init() ;

  Arccore::UniqueArray<Arccore::Int64> uids(local_nrows+nb_ghosts) ;
  Arccore::UniqueArray<Arccore::Integer> owners(local_nrows+nb_ghosts) ;
  for(int i=0;i<local_nrows;++i)
  {
    uids[i] = row_uids[i] ;
    owners[i] = my_rank ;
    m_uid2lid[row_uids[i]] = i ;
  }
  for(int i=0;i<nb_ghosts;++i)
  {
    uids[local_nrows+i] = ghost_uids[i] ;
    owners[local_nrows+i] = ghost_owners[i] ;
    m_uid2lid[ghost_uids[i]] = local_nrows+i ;
  }

  
  //m_item_family.reset(new Alien::DefaultAbstractFamily(uids,owners,mng->parallelMng())) ;
  m_item_family.reset(new Alien::AbstractItemFamily(uids,owners,mng->parallelMng())) ;


  m_index_set = m_index_manager->buildScalarIndexSet("U", *m_item_family,0);

  m_index_manager->prepare() ;

  auto global_size = m_index_manager->globalSize();
  auto local_size = m_index_manager->localSize();


  m_allUIndex = m_index_manager->getIndexes(m_index_set);

  m_space.reset(new Alien::Space(global_size, "SystemSpace"));


  m_matrix_dist.reset(new Alien::MatrixDistribution(global_size, global_size, local_size, mng->parallelMng()));
  m_vector_dist.reset(new Alien::VectorDistribution(global_size, local_size, mng->parallelMng()));

  m_A.reset(new Alien::Matrix(*m_matrix_dist));
  m_x.reset(new Alien::Vector(*m_vector_dist));
  m_b.reset(new Alien::Vector(*m_vector_dist));

}

void
AlienManager::LinearSystem::
defineMatrixProfile(int local_nrows,
                    uid_type* row_uids,
                    int* row_offset,
                    uid_type* col_uids)
{

  //auto* mng = AlienManager::instance() ;
  //auto* pm = mng->parallelMng() ;
  //auto* tm = mng->traceMng() ;

  Alien::MatrixProfiler profiler(*m_A);
  ///////////////////////////////////////////////////////////////////////////
  //
  // DEFINE PROFILE
  //
  for(int irow=0;irow<local_nrows;++irow)
  {
    int row_index = m_allUIndex[m_uid2lid[row_uids[irow]]] ;
    for(int k=row_offset[irow];k<row_offset[irow+1];++k)
    {
      int col_index = m_allUIndex[m_uid2lid[col_uids[k]]] ;
      profiler.addMatrixEntry(row_index, col_index);
    }
  }
}


void
AlienManager::LinearSystem::
setMatrixValues(int local_nrows,
                uid_type* row_uids,
                int* row_offset,
                uid_type* col_uids,
                double* values)
{

  //auto* mng = AlienManager::instance() ;
  //auto* pm = mng->parallelMng() ;
  //auto* tm = mng->traceMng() ;

  Alien::ProfiledMatrixBuilder builder(*m_A, Alien::ProfiledMatrixOptions::eResetValues);

  for(int irow=0;irow<local_nrows;++irow)
  {
    uid_type row_uid = row_uids[irow] ;
    int row_lid = m_uid2lid[row_uid] ;
    int row_index = m_allUIndex[row_lid] ;
    for(int k=row_offset[irow];k<row_offset[irow+1];++k)
    {
      uid_type col_uid = col_uids[k] ;
      int col_lid = m_uid2lid[col_uid] ;
      int col_index = m_allUIndex[col_lid] ;
      //tm->pinfo()<<pm->commRank()<<" addMatrixEntry("<<row_index<<","<<col_index<<")+="<<values[k];
      builder(row_index,col_index) += values[k];
    }
  }
}


void
AlienManager::LinearSystem::
setRHSValues(int local_nrows,
             uid_type* row_uids,
             double const* values)
{

  Alien::VectorWriter v(*m_b);
  for (int irow = 0; irow < local_nrows; ++irow)
  {
    uid_type row_uid = row_uids[irow] ;
    int row_lid = m_uid2lid[row_uid] ;
    int row_index = m_allUIndex[row_lid] ;
    v[row_index] = values[irow];
  }
}

void
AlienManager::LinearSystem::
getSolutionValues(int local_nrows,
                  uid_type* row_uids,
                  double* values)
{

  Alien::VectorReader v(*m_x);
  for (int irow = 0; irow < local_nrows; ++irow)
  {
    uid_type row_uid = row_uids[irow] ;
    int row_lid = m_uid2lid[row_uid] ;
    int row_index = m_allUIndex[row_lid] ;
    values[irow] = v[row_index] ;
  }
}


void
AlienManager::LinearSolver::
init(int argc, const char** argv)
{
  using namespace boost::program_options;
  options_description generic("Generic options");
  generic.add_options()("help", "produce help")
      ("solver-package", value<std::string>()->default_value("petsc"), "solver package name")
      ("max-iter", value<int>()->default_value(1000), "max iterations")
      ("tol", value<double>()->default_value(1.e-10), "solver tolerance")
      ("sym", value<int>()->default_value(1), "0->nsym, 1->sym")
      ("output-level", value<int>()->default_value(0), "output level");

  options_description parse_command_line_desc ;
  parse_command_line_desc.add(generic) ;
  Alien::SolverFactory::add_options("petsc",parse_command_line_desc) ;
  Alien::SolverFactory::add_options("hypre",parse_command_line_desc) ;
  Alien::SolverFactory::add_options("mtl4",parse_command_line_desc) ;
  Alien::SolverFactory::add_options("ifpsolver",parse_command_line_desc) ;
  Alien::SolverFactory::add_options("htssolver",parse_command_line_desc) ;
  Alien::SolverFactory::add_options("mcgsolver",parse_command_line_desc) ;
  Alien::SolverFactory::add_options("trilinos",parse_command_line_desc) ;

  variables_map vm;
  store(parse_command_line(argc,argv,parse_command_line_desc), vm);
  notify(vm);


  auto* mng = AlienManager::instance() ;

  auto* pm = mng->parallelMng() ;
  auto* tm = mng->traceMng();

  if (vm.count("help")) {
    tm->fatal() << parse_command_line_desc ;
  }

  std::string solver_package = vm["solver-package"].as<std::string>();

  tm->info() << "Try to create solver-package : " << solver_package;
  m_linear_solver.reset(Alien::SolverFactory::create(solver_package,vm,pm)) ;
  if(m_linear_solver.get()==nullptr)
    tm->fatal() << "*** package " << solver_package << " not available!";
}


void
AlienManager::LinearSolver::
init(std::string const& configfile)
{
  auto* mng = AlienManager::instance() ;

  auto* pm = mng->parallelMng() ;
  auto* tm = mng->traceMng();

  std::cout<<"CONFIG FILE : "<<configfile<<std::endl ;

  // Short alias for this namespace
  namespace pt = boost::property_tree;

  // Create a root
  pt::ptree root;

  // Load the json file in this ptree
  pt::read_json(configfile, root);



  std::string solver_package = root.get<std::string>("solver-package");

  tm->info() << "Try to create solver-package : " << solver_package;

  pt::ptree config = root.get_child("config") ;
  m_linear_solver.reset(Alien::SolverFactory::create(solver_package,config,pm)) ;
  if(m_linear_solver.get()==nullptr)
    tm->fatal() << "*** package " << solver_package << " not available!";
}


int
AlienManager::LinearSolver::
solve(Alien::Matrix const& A, Alien::Vector const& B, Alien::Vector& X)
{

  auto* mng = AlienManager::instance() ;
  auto* pm = mng->parallelMng() ;
  int comm_rank = pm->commRank() ;
  auto* tm = mng->traceMng() ;
  assert(m_linear_solver.get()!=nullptr) ;
  m_linear_solver->init() ;
  m_linear_solver->solve(A, B, X);
  const auto& status = m_linear_solver->getStatus();
  if (status.succeeded)
  {
    if (comm_rank == 0) {
      tm->info() << "Solver succeed   " << status.succeeded ;
      tm->info() << "             residual " << status.residual ;
      tm->info() << "             nb iters  " << status.iteration_count ;
      tm->info() << "             error      " << status.error ;
    }
    return 0 ;
  }
  else
  {
    tm->info() << "Solver not succeed   " << status.succeeded ;
    tm->info() << "           residual " << status.residual ;
    tm->info() << "           nb iters " << status.iteration_count ;
    tm->info() << "           error    " << status.error ;
    return 1 ;
  }
}

void
AlienManager::LinearSolver::
getStatus(int* code, double* residual, int* num_iterations)
{

  const auto& status = m_linear_solver->getStatus();
  if (status.succeeded)
  {
    *code = 0 ;
    *residual = status.residual;
    *num_iterations = status.iteration_count ;
  }
  else
    *code = status.error ;
}
AlienManager* AlienManager::m_singleton = nullptr ;

extern "C" {

  #include "alien/c/alienc.h"

  int ALIEN_init([[maybe_unused]] int argc,[[maybe_unused]]  char** argv)
  {
    AlienManager::initialize() ;
    return 0 ;
  }

  int ALIEN_create_linear_system(MPI_Comm comm)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    return alien_mng->createNewLinearSystem(comm) ;
  }

  int ALIEN_destroy_linear_system(int system_id)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    return alien_mng->destroyLinearSystem(system_id) ;
  }


  int ALIEN_init_linear_system(int system_id,
                               int global_nrows,
                               int local_nrows,
                               uid_type* row_uids,
                               int nb_ghosts,
                               uid_type* ghost_uids,
                               int* ghost_owners)
  {

    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    AlienManager::LinearSystem* system = alien_mng->getLinearSystem(system_id) ;
    assert(system!=nullptr) ;
    system->init(global_nrows,
                 local_nrows,
                 row_uids,
                 nb_ghosts,
                 ghost_uids,
                 ghost_owners) ;
    return 0 ;
  }

  int ALIEN_define_matrix_profile(int system_id,
                                  int local_nrows,
                                  uid_type* row_uids,
                                  int* row_offset,
                                  uid_type* col_uids)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    AlienManager::LinearSystem* system = alien_mng->getLinearSystem(system_id) ;
    assert(system!=nullptr) ;
    system->defineMatrixProfile(local_nrows,
                                row_uids,
                                row_offset,
                                col_uids) ;
    return 0 ;
  }

  int ALIEN_set_matrix_values(int system_id,
                              int local_nrows,
                              uid_type* row_uids,
                              int* row_offset,
                              uid_type* col_uids,
                              double* values)
  {

    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    AlienManager::LinearSystem* system = alien_mng->getLinearSystem(system_id) ;
    assert(system!=nullptr) ;
    system->setMatrixValues(local_nrows,
                            row_uids,
                            row_offset,
                            col_uids,
                            values) ;
    return 0 ;
  }

  int ALIEN_set_rhs_values(int system_id,
                           int local_nrows,
                           uid_type* row_uids,
                           double const* values)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    AlienManager::LinearSystem* system = alien_mng->getLinearSystem(system_id) ;
    assert(system!=nullptr) ;
    system->setRHSValues(local_nrows,row_uids,values) ;
    return 0 ;
  }

  int ALIEN_get_solution_values(int system_id,
                                int local_nrows,
                                uid_type* row_uids,
                                double* values)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    AlienManager::LinearSystem* system = alien_mng->getLinearSystem(system_id) ;
    assert(system!=nullptr) ;
    system->getSolutionValues(local_nrows,row_uids,values) ;
    return 0 ;
  }


  int ALIEN_create_parameter_system()
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    return alien_mng->createNewParamSystem() ;
  }


  int ALIEN_destroy_parameter_system(int param_system_id)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    alien_mng->destroyParamSystem(param_system_id) ;
    return 0 ;
  }

  void ALIEN_set_parameter_string_value(int param_system_id,
                                        const char* key,
                                        const char* value)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* param_system = alien_mng->getParamSystem(param_system_id) ;
    assert(param_system!=nullptr) ;
    param_system->setParam(std::string(key),std::string(value)) ;
  }

  void ALIEN_set_parameter_integer_value(int param_system_id,
                                         const char* key,
                                         int value)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* param_system = alien_mng->getParamSystem(param_system_id) ;
    assert(param_system!=nullptr) ;
    param_system->setParam(std::string(key),value) ;
  }

  void ALIEN_set_parameter_double_value(int param_system_id,
                                        const char* key,
                                        double value)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* param_system = alien_mng->getParamSystem(param_system_id) ;
    assert(param_system!=nullptr) ;
    param_system->setParam(std::string(key),value) ;
  }

  int ALIEN_create_solver(MPI_Comm comm,const char* config_file)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    return alien_mng->createNewLinearSolver(comm,config_file) ;
  }


  int ALIEN_init_solver(int solver_id,int argc, const char** argv)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* solver = alien_mng->getLinearSolver(solver_id) ;
    solver->init(argc,argv) ;
    return 0 ;
  }


  int ALIEN_init_solver_with_configfile(int solver_id,const char* path)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* solver = alien_mng->getLinearSolver(solver_id) ;
    std::string configfile(path) ;
    solver->init(configfile) ;
    return 0 ;
  }

  int ALIEN_init_solver_with_parameters(int solver_id,int param_system_id)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* solver = alien_mng->getLinearSolver(solver_id) ;
    assert(solver!=nullptr) ;

    auto* param_system = alien_mng->getParamSystem(param_system_id) ;
    assert(param_system!=nullptr) ;
    auto command_line = param_system->commandLine() ;
    solver->init(command_line.size(),command_line.data()) ;
    return 0 ;
  }

  int ALIEN_destroy_solver(int solver_id)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    alien_mng->destroyLinearSolver(solver_id) ;
    return 0 ;
  }

  int ALIEN_solve(int solver_id, int system_id)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* solver = alien_mng->getLinearSolver(solver_id) ;
    assert(solver!=nullptr) ;
    auto* system = alien_mng->getLinearSystem(system_id) ;
    assert(system!=nullptr) ;
    return solver->solve(system->getA(),system->getB(),system->getX()) ;
  }

  int ALIEN_get_solver_status(int solver_id, ALIEN_Solver_Status* status)
  {
    auto* alien_mng = AlienManager::instance() ;
    assert(alien_mng!=nullptr) ;
    auto* solver = alien_mng->getLinearSolver(solver_id) ;
    solver->getStatus(&status->code, &status->residual, &status->num_iterations) ;
    return 0 ;
  }

  int ALIEN_finalize()
  {
    AlienManager::finalize() ;
    return 0 ;
  }

}


