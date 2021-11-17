#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>

#include <MueLu.hpp>
#ifdef HAVE_MUELU_AMGX
#include <amgx_capi.h>
#include <MueLu_AMGX_Setup.hpp>
#endif

#ifdef _WIN32
#include "windows.h"
#define amgx_libopen(path) (void*)(LoadLibrary(path))
#define amgx_liblink(handle, symbol) GetProcAddress((HMODULE)(handle), symbol)
#define amgx_libclose(handle) FreeLibrary((HMODULE)(handle))
#endif

#ifdef __unix__

#include <dlfcn.h>
#include <unistd.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#define amgx_libopen(path) dlopen(path, RTLD_LAZY)
#define amgx_liblink(handle, symbol) dlsym(handle, symbol)
#define amgx_libclose(handle) dlclose(handle)
#endif
/*---------------------------------------------------------------------------*/
BEGIN_TRILINOSINTERNAL_NAMESPACE

bool TrilinosInternal::m_is_initialized = false;
bool TrilinosInternal::m_amgx_is_initialized = false;
int TrilinosInternal::m_nb_threads = 1;
std::size_t TrilinosInternal::m_nb_hyper_threads = 1;
std::size_t TrilinosInternal::m_mpi_core_id_offset = 0;
std::string TrilinosInternal::m_execution_space = "Serial";
std::unique_ptr<const Teuchos::Comm<int>> TrilinosInternal::m_trilinos_comm;

const std::string TrilinosInternal::Node<BackEnd::tag::tpetraserial>::name =
    "tpetraserial";
const std::string
    TrilinosInternal::Node<BackEnd::tag::tpetraserial>::execution_space_name = "Serial";

#ifdef KOKKOS_ENABLE_OPENMP
const std::string TrilinosInternal::Node<BackEnd::tag::tpetraomp>::name = "tpetraomp";
const std::string TrilinosInternal::Node<BackEnd::tag::tpetraomp>::execution_space_name =
    "OpenMP";
#endif

#ifdef KOKKOS_ENABLE_THREADS
const std::string TrilinosInternal::Node<BackEnd::tag::tpetrapth>::name = "tpetrapth";
const std::string TrilinosInternal::Node<BackEnd::tag::tpetrapth>::execution_space_name =
    "Threads";
#endif

#ifdef KOKKOS_ENABLE_CUDA
const std::string TrilinosInternal::Node<BackEnd::tag::tpetracuda>::name = "tpetracuda";
const std::string TrilinosInternal::Node<BackEnd::tag::tpetracuda>::execution_space_name =
    "Cuda";
#endif

template <>
int
TrilinosInternal::getEnv<int>(std::string const& key, int default_value)
{
  const char* env_str = ::getenv(key.c_str());
  if (env_str)
    return atoi(env_str);
  return default_value;
}

template <>
std::string
TrilinosInternal::getEnv<std::string>(std::string const& key, std::string default_value)
{
  const char* env_str = ::getenv(key.c_str());
  if (env_str)
    return std::string(env_str);
  return default_value;
}

void
TrilinosInternal::initialize(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::string const& execution_space, int nb_threads, bool use_amgx)
{
  if (m_is_initialized)
    return;
  m_nb_threads = getEnv<int>("TRILINOS_NUM_THREADS", 1);
  if (nb_threads > 0)
    m_nb_threads = nb_threads;

  if (parallel_mng && (parallel_mng->commSize() > 1)) {
    auto* mpi_mng =
        dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(parallel_mng);
    const MPI_Comm* comm = static_cast<const MPI_Comm*>(mpi_mng->getMPIComm());
    m_trilinos_comm.reset(new Teuchos::MpiComm<int>(*comm));
  }
  m_execution_space = getEnv<std::string>("KOKKOS_EXECUTION_SPACE", "Serial");
  if (execution_space.compare("Undefined") != 0)
    m_execution_space = execution_space;

  if (m_execution_space.compare("Serial") == 0) {
    Kokkos::Serial::initialize();
  }
#ifdef KOKKOS_ENABLE_OPENMP
  if (m_execution_space.compare("OpenMP") == 0) {
    Kokkos::OpenMP::initialize(m_nb_threads);
    std::cout << "OMP NUM THREADS : " << omp_get_num_threads() << std::endl;
  }
#endif
#ifdef KOKKOS_ENABLE_THREADS
  if (m_execution_space.compare("Threads") == 0) {
    Kokkos::Threads::initialize(m_nb_threads);
  }
#endif
#ifdef KOKKOS_ENABLE_CUDA
  if (m_execution_space.compare("Cuda") == 0) {
    Kokkos::InitArguments args;
    args.num_threads = m_nb_threads;
    args.num_numa = 1;
    args.device_id = 0;
// Kokkos::initialize(args) ;
// Kokkos::Cuda::initialize() ;

#ifdef HAVE_MUELU_AMGX
    if (use_amgx && !m_amgx_is_initialized) {
      void* lib_handle = NULL;
// open the library
#ifdef _WIN32
      lib_handle = amgx_libopen("amgxsh.dll");
#else
      lib_handle = amgx_libopen("libamgxsh.so");
#endif

      if (lib_handle == NULL) {
        std::cerr << "ERROR: can not load the library" << std::endl;
        return;
      }

      // load all the routines
      if (amgx_liblink_all(lib_handle) == 0) {
        amgx_libclose(lib_handle);
        std::cerr << "ERROR: corrupted library loaded" << std::endl;
        return;
      }

      AMGX_SAFE_CALL(AMGX_initialize());
      AMGX_SAFE_CALL(AMGX_initialize_plugins());
      m_amgx_is_initialized = true;
    }
#endif
  }
#endif
  m_is_initialized = true;
}

void
TrilinosInternal::initMPIEnv(MPI_Comm comm)
{
}

void
TrilinosInternal::finalize()
{
#ifdef HAVE_MUELU_AMGX
  if (m_amgx_is_initialized) {
    // Finalize AMGX
    MueLu::MueLu_AMGX_finalize_plugins();
    MueLu::MueLu_AMGX_finalize();
    m_amgx_is_initialized = false;
  }
#endif
  m_is_initialized = false;
}

END_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
