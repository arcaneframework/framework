#include <alien/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/Kernels/Trilinos/DataStructure/TrilinosInternal.h>


/*---------------------------------------------------------------------------*/
BEGIN_TRILINOSINTERNAL_NAMESPACE

bool                                      TrilinosInternal::m_is_initialized     = false  ;
int                                       TrilinosInternal::m_nb_threads         = 1  ;
std::size_t                               TrilinosInternal::m_nb_hyper_threads   = 1;
std::size_t                               TrilinosInternal::m_mpi_core_id_offset = 0 ;
std::string                               TrilinosInternal::m_execution_space  = "Serial";
std::unique_ptr<const Teuchos::Comm<int> > TrilinosInternal::m_trilinos_comm;


const std::string TrilinosInternal::Node<BackEnd::tag::tpetraserial>::name = "tpetraserial";
const std::string TrilinosInternal::Node<BackEnd::tag::tpetraserial>::execution_space_name = "Serial";

#ifdef KOKKOS_ENABLE_OPENMP
const std::string TrilinosInternal::Node<BackEnd::tag::tpetraomp>::name = "tpetraomp";
const std::string TrilinosInternal::Node<BackEnd::tag::tpetraomp>::execution_space_name = "OpenMP";

#endif

#ifdef KOKKOS_ENABLE_THREADS
const std::string TrilinosInternal::Node<BackEnd::tag::tpetrapth>::name = "tpetrapth";
const std::string TrilinosInternal::Node<BackEnd::tag::tpetrapth>::execution_space_name = "Threads";

#endif

template<>
int TrilinosInternal::getEnv<int>(std::string const& key,int default_value)
{
  const char* env_str = ::getenv(key.c_str());
  if (env_str)
    return atoi(env_str);
  return default_value ;
}

template<>
std::string TrilinosInternal::getEnv<std::string>(std::string const& key,std::string default_value)
{
  const char* env_str = ::getenv(key.c_str());
  if (env_str)
    return std::string(env_str);
  return default_value ;
}

void TrilinosInternal::initialize(IParallelMng* parallel_mng,std::string const& execution_space, int nb_threads)
{
  if(m_is_initialized)
    return ;
  m_nb_threads = getEnv<int>("TRILINOS_NUM_THREADS",1) ;
  if(nb_threads>0)
    m_nb_threads = nb_threads ;

  if(parallel_mng && parallel_mng->isParallel())
  {
    MPI_Comm* comm = static_cast<MPI_Comm*>(parallel_mng->getMPICommunicator()) ;
    m_trilinos_comm.reset(new Teuchos::MpiComm<int> (*comm));
  }
  m_execution_space = getEnv<std::string>("KOKKOS_EXECUTION_SPACE","Serial") ;
  if(execution_space.compare("Undefined")!=0)
    m_execution_space = execution_space ;

  if(m_execution_space.compare("Serial")==0)
  {
    Kokkos::Serial::initialize() ;
  }
  if(m_execution_space.compare("OpenMP")==0)
  {
    Kokkos::OpenMP::initialize(m_nb_threads) ;
  }
  if(m_execution_space.compare("Threads")==0)
  {
    Kokkos::Threads::initialize(m_nb_threads) ;
  }

  m_is_initialized = true ;
}

void TrilinosInternal::initMPIEnv(MPI_Comm comm)
{

}

void TrilinosInternal::finalize()
{
  m_is_initialized = false ;
}

END_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
