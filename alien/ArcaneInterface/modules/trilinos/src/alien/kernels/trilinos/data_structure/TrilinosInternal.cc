
#include <alien/kernels/trilinos/TrilinosBackEnd.h>


#include <Teuchos_ParameterList.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Distributor.hpp>
#include <Tpetra_HashTable.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_Import_Util.hpp>

#include <MueLu.hpp>
#ifdef HAVE_MUELU_AMGX
//#define USE_DYNAMIC_AMGXLIB
#ifdef USE_DYNAMIC_AMGXLIB
#include <amgx_capi.h>
#endif
#include <MueLu_AMGX_Setup.hpp>
#endif

#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>

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


/* print callback (could be customized) */
void print_callback(const char *msg, int length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { printf("%s", msg); }
}

/*---------------------------------------------------------------------------*/
BEGIN_TRILINOSINTERNAL_NAMESPACE

bool TrilinosInternal::m_is_initialized            = false;
bool TrilinosInternal::m_amgx_is_initialized       = false;
int TrilinosInternal::m_nb_threads                 = 1;
std::size_t TrilinosInternal::m_nb_hyper_threads   = 1;
std::size_t TrilinosInternal::m_mpi_core_id_offset = 0;
std::string TrilinosInternal::m_execution_space    = "Serial";
std::unique_ptr<const Teuchos::Comm<int>> TrilinosInternal::m_trilinos_comm;


#ifdef KOKKOS_ENABLE_SERIAL
const std::string TrilinosInternal::Node<BackEnd::tag::tpetraserial>::name =
    "tpetraserial";
const std::string
    TrilinosInternal::Node<BackEnd::tag::tpetraserial>::execution_space_name = "Serial";
#endif

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

#ifdef KOKKOS_ENABLE_SERIAL
  if (m_execution_space.compare("Serial") == 0) {
    //Kokkos::Serial::initialize();
    Kokkos::initialize() ;
  }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  if (m_execution_space.compare("OpenMP") == 0) {
#if (TRILINOS_MAJOR_VERSION >= 13)
    Kokkos::InitArguments args;
    args.num_threads = m_nb_threads;
    Kokkos::initialize(args) ;
#else
    Kokkos::OpenMP::initialize(m_nb_threads);
#endif
    std::cout << "OMP NUM THREADS : " << omp_get_max_threads() << " " << m_nb_threads << std::endl;
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
    Kokkos::initialize(args) ;
    // Kokkos::Cuda::initialize() ;
  }
#endif
  m_is_initialized = true;
}

#ifdef HAVE_MUELU_AMGX
void initAMGX(AMGXEnv& amgx_env)
{
    typedef int local_index_type ;
    typedef int global_index_type ;
    typedef TrilinosInternal::Node<BackEnd::tag::tpetracuda>::type node_type ;

    std::cout<<"INIT AMGX ENV FROM MueLU MATRIX"<<std::endl ;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcpFromRef; // Make a "weak" RCP from a reference.
    using Teuchos::ArrayRCP;
    using Teuchos::Array ;
    using Teuchos::ArrayView ;
    if (!TrilinosInternal::m_amgx_is_initialized) 
    {
#ifdef USE_DYNAMIC_AMGXLIB
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
#endif

      AMGX_SAFE_CALL(AMGX_initialize());
      AMGX_SAFE_CALL(AMGX_initialize_plugins());

      std::cout<<"TEST AMGX LIBRARY LOAD STEP"<<std::endl;

      //library handles
      //AMGX_config_handle cfg;
      //AMGX_resources_handle rsrc;

      /* system */
      AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
      AMGX_SAFE_CALL(AMGX_install_signal_handler());

      std::cout<<"TEST AMGX LIBRARY LOAD STEP"<<std::endl;

      /* create config */
      if(amgx_env.m_config_file.compare("undefined")!=0)
      {
        std::cout<<"CREATE AMGX CONFIG FROM FILE : "<<amgx_env.m_config_file<<std::endl ;
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx_env.m_config, amgx_env.m_config_file.c_str()));
      }
      else
      {
        std::cout<<"CREATE AMGX CONFIG FROM STRING : "<<amgx_env.m_config_string<<std::endl ;
        AMGX_SAFE_CALL(AMGX_config_create(&amgx_env.m_config, amgx_env.m_config_string.c_str()));
      }
      //AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx_env.m_config, "exception_handling=1"));
      std::cout<<"CREATE AMGX RESOURCES"<<std::endl ;
      AMGX_resources_create_simple(&amgx_env.m_resources,amgx_env.m_config);
      AMGX_Mode mode = AMGX_mode_dDDI;
      AMGX_solver_create(&amgx_env.m_solver, amgx_env.m_resources, mode,amgx_env.m_config);
      AMGX_matrix_create(&amgx_env.m_A,      amgx_env.m_resources, mode);
      AMGX_vector_create(&amgx_env.m_X,      amgx_env.m_resources, mode);
      AMGX_vector_create(&amgx_env.m_Y,      amgx_env.m_resources, mode);


      std::cout<<"AMGX CONFIG LIBRARY LOAD STEP SUCCESFULL"<<std::endl;
      TrilinosInternal::m_amgx_is_initialized = true ;
    }
}
void solveAMGX(AMGXEnv& amgx_env,
	       double* x, 
	       double* y)
{
   //std::cout<<"AMGX VECTOR UPLOAD FROM HOST"<<std::endl ;
   AMGX_vector_upload(amgx_env.m_X, amgx_env.m_N, 1, x);
   AMGX_vector_upload(amgx_env.m_Y, amgx_env.m_N, 1, y);
   //std::cout<<"AMGX VECTOR UPLOAD FROM HOST OK"<<std::endl ;
   //std::cout<<"TEST SOLVER : "<<std::endl ;
   AMGX_solver_solve(amgx_env.m_solver, amgx_env.m_X, amgx_env.m_Y);
   //AMGX_SOLVE_STATUS status;
   //AMGX_solver_get_status(amgx_env.m_solver, &status);
   AMGX_vector_download(amgx_env.m_Y,y);
   //std::cout<<"AFTER TEST SOLVER : "<<std::endl ;
}
#endif

#ifdef KOKKOS_ENABLE_CUDA
#ifdef HAVE_MUELU_AMGX
void initAMGX(AMGXEnv& amgx_env,
	      Tpetra::CrsMatrix<double, int, int,  TrilinosInternal::Node<BackEnd::tag::tpetracuda>::type> const& inA)
{
    typedef int local_index_type ;
    typedef int global_index_type ;
    typedef TrilinosInternal::Node<BackEnd::tag::tpetracuda>::type node_type ;

    std::cout<<"INIT AMGX ENV FROM MueLU MATRIX"<<std::endl ;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::rcpFromRef; // Make a "weak" RCP from a reference.
    using Teuchos::ArrayRCP;
    using Teuchos::Array ;
    using Teuchos::ArrayView ;
    if (!TrilinosInternal::m_amgx_is_initialized) 
    {
#ifdef USE_DYNAMIC_AMGXLIB
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
#endif

      AMGX_SAFE_CALL(AMGX_initialize());
      AMGX_SAFE_CALL(AMGX_initialize_plugins());

      std::cout<<"TEST AMGX LIBRARY LOAD STEP"<<std::endl;

      //library handles
      //AMGX_config_handle cfg;
      //AMGX_resources_handle rsrc;

      /* system */
      AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
      AMGX_SAFE_CALL(AMGX_install_signal_handler());

      std::cout<<"TEST AMGX LIBRARY LOAD STEP"<<std::endl;

      /* create config */
      if(amgx_env.m_config_file.compare("undefined")!=0)
      {
        std::cout<<"CREATE AMGX CONFIG FROM FILE : "<<amgx_env.m_config_file<<std::endl ;
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&amgx_env.m_config, amgx_env.m_config_file.c_str()));
      }
      else
      {
        std::cout<<"CREATE AMGX CONFIG FROM STRING : "<<amgx_env.m_config_string<<std::endl ;
        AMGX_SAFE_CALL(AMGX_config_create(&amgx_env.m_config, amgx_env.m_config_string.c_str()));
      }
      //AMGX_SAFE_CALL(AMGX_config_add_parameters(&amgx_env.m_config, "exception_handling=1"));
      std::cout<<"CREATE AMGX RESOURCES"<<std::endl ;
      AMGX_resources_create_simple(&amgx_env.m_resources,amgx_env.m_config);
      AMGX_Mode mode = AMGX_mode_dDDI;
      AMGX_solver_create(&amgx_env.m_solver, amgx_env.m_resources, mode,amgx_env.m_config);
      AMGX_matrix_create(&amgx_env.m_A,      amgx_env.m_resources, mode);
      AMGX_vector_create(&amgx_env.m_X,      amgx_env.m_resources, mode);
      AMGX_vector_create(&amgx_env.m_Y,      amgx_env.m_resources, mode);


      std::cout<<"AMGX CONFIG LIBRARY LOAD STEP SUCCESFULL"<<std::endl;
      TrilinosInternal::m_amgx_is_initialized = true ;
    }

      RCP<const Teuchos::Comm<int> > comm = inA.getRowMap()->getComm();
      int numProcs = comm->getSize();
      int myRank   = comm->getRank();
      std::vector<int> amgx2muelu;

      // Construct AMGX communication pattern
      if (numProcs > 1) 
      {
        RCP<const Tpetra::Import<local_index_type,global_index_type,node_type> > importer = inA.getCrsGraph()->getImporter();

        TEUCHOS_TEST_FOR_EXCEPTION(importer.is_null(), MueLu::Exceptions::RuntimeError, "The matrix A has no Import object.");

        Tpetra::Distributor distributor = importer->getDistributor();

	Teuchos::Array<int> sendRanks = distributor.getProcsTo();
	Teuchos::Array<int> recvRanks = distributor.getProcsFrom();

        std::sort(sendRanks.begin(), sendRanks.end());
        std::sort(recvRanks.begin(), recvRanks.end());

        bool match = true;
        if (sendRanks.size() != recvRanks.size()) {
          match = false;
        } else {
          for (int i = 0; i < sendRanks.size(); i++) {
            if (recvRanks[i] != sendRanks[i])
              match = false;
              break;
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(!match, MueLu::Exceptions::RuntimeError, "AMGX requires that the processors that we send to and receive from are the same. "
                                   "This is not the case: we send to {" << sendRanks << "} and receive from {" << recvRanks << "}");

        int        num_neighbors = sendRanks.size();  // does not include the calling process
        const int* neighbors     = &sendRanks[0];

        // Later on, we'll have to organize the send and recv data by PIDs,
        // i.e, a vector V of vectors, where V[i] is PID i's vector of data.
        // Hence we need to be able to quickly look up  an array index
        // associated with each PID.
        Tpetra::Details::HashTable<int,int> hashTable(3*num_neighbors);
        for (int i = 0; i < num_neighbors; i++)
          hashTable.add(neighbors[i], i);

        // Get some information out
        ArrayView<const int> exportLIDs = importer->getExportLIDs();
        ArrayView<const int> exportPIDs = importer->getExportPIDs();
        Array<int> importPIDs;
        Tpetra::Import_Util::getPids(*importer, importPIDs, true/* make local -1 */);

        // Construct the reordering for AMGX as in AMGX_matrix_upload_all documentation
        //RCP<const Tpetra::Map> rowMap = inA.getRowMap();
        //RCP<const Tpetra::Map> colMap = inA.getColMap();

        int N = inA.getRowMap()->getNodeNumElements(), Nc = inA.getColMap()->getNodeNumElements();
        amgx_env.m_muelu2amgx.resize(Nc, -1);

        int numUniqExports = 0;
        for (int i = 0; i < exportLIDs.size(); i++)
          if (amgx_env.m_muelu2amgx[exportLIDs[i]] == -1) {
            numUniqExports++;
            amgx_env.m_muelu2amgx[exportLIDs[i]] = -2;
          }

        int localOffset = 0, exportOffset = N - numUniqExports;
        // Go through exported LIDs and put them at the end of LIDs
        for (int i = 0; i < exportLIDs.size(); i++)
          if (amgx_env.m_muelu2amgx[exportLIDs[i]] < 0) // exportLIDs are not unique
            amgx_env.m_muelu2amgx[exportLIDs[i]] = exportOffset++;
        // Go through all non-export LIDs, and put them at the beginning of LIDs
        for (int i = 0; i < N; i++)
          if (amgx_env.m_muelu2amgx[i] == -1)
            amgx_env.m_muelu2amgx[i] = localOffset++;
        // Go through the tail (imported LIDs), and order those by neighbors
        int importOffset = N;
        for (int k = 0; k < num_neighbors; k++)
          for (int i = 0; i < importPIDs.size(); i++)
            if (importPIDs[i] != -1 && hashTable.get(importPIDs[i]) == k)
             amgx_env.m_muelu2amgx[i] = importOffset++;

        amgx2muelu.resize(amgx_env.m_muelu2amgx.size());
        for (int i = 0; i < (int)amgx_env.m_muelu2amgx.size(); i++)
          amgx2muelu[amgx_env.m_muelu2amgx[i]] = i;

        // Construct send arrays
        std::vector<std::vector<int> > sendDatas (num_neighbors);
        std::vector<int>               send_sizes(num_neighbors, 0);
        for (int i = 0; i < exportPIDs.size(); i++) {
          int index = hashTable.get(exportPIDs[i]);
          sendDatas [index].push_back(amgx_env.m_muelu2amgx[exportLIDs[i]]);
          send_sizes[index]++;
        }
        // FIXME: sendDatas must be sorted (based on GIDs)

        std::vector<const int*> send_maps(num_neighbors);
        for (int i = 0; i < num_neighbors; i++)
          send_maps[i] = &(sendDatas[i][0]);

        // Debugging
        //        printMaps(comm, sendDatas, amgx2muelu, neighbors, *importer->getTargetMap(), "send_map_vector");

        // Construct recv arrays
        std::vector<std::vector<int> > recvDatas (num_neighbors);
        std::vector<int>               recv_sizes(num_neighbors, 0);
        for (int i = 0; i < importPIDs.size(); i++)
          if (importPIDs[i] != -1) {
            int index = hashTable.get(importPIDs[i]);
            recvDatas [index].push_back(amgx_env.m_muelu2amgx[i]);
            recv_sizes[index]++;
        }
        // FIXME: recvDatas must be sorted (based on GIDs)

        std::vector<const int*> recv_maps(num_neighbors);
        for (int i = 0; i < num_neighbors; i++)
          recv_maps[i] = &(recvDatas[i][0]);

        // Debugging
        //        printMaps(comm, recvDatas, amgx2muelu, neighbors, *importer->getTargetMap(), "recv_map_vector");

        AMGX_SAFE_CALL(AMGX_matrix_comm_from_maps_one_ring(amgx_env.m_A, 1, num_neighbors, neighbors, &send_sizes[0], &send_maps[0], &recv_sizes[0], &recv_maps[0]));

        //AMGX_vector_bind(X_, A_);
        //AMGX_vector_bind(Y_, A_);
      }


      RCP<Teuchos::Time> matrixTransformTimer = Teuchos::TimeMonitor::getNewTimer("MueLu: AMGX: transform matrix");
      matrixTransformTimer->start();
      amgx_env.m_N      = inA.getNodeNumRows();
      amgx_env.m_nnz    = inA.getNodeNumEntries();
      int n             = amgx_env.m_N ;
      int nnz           = amgx_env.m_nnz ;
      std::cout<<"NUMROWS NNZ : "<<n<<" "<<nnz<<std::endl ;

      ArrayRCP<const size_t> ia_s;
      ArrayRCP<const int>    ja_s;
      ArrayRCP<const double> a_s;
      inA.getAllValues(ia_s, ja_s, a_s);
      std::cout<<"IA SIZE : "<<ia_s.size()<<std::endl ;
      std::cout<<"JA SIZE : "<<ja_s.size()<<std::endl ;
      std::cout<<"A  SIZE : "<<a_s.size()<<std::endl ;
      std::vector<int>   ia(amgx_env.m_N+1);
      std::vector<int>   ja(amgx_env.m_nnz);
      std::vector<double> a(amgx_env.m_nnz);


      for(int i=0;i<n;++i)
      {
        ia[i] = Teuchos::as<int>(ia_s[i]);
        for(int k=ia_s[i];k<ia_s[i+1];++k)
        {
	  ja[k]=ja_s[k] ;
	  a[k]=a_s[k] ;
	  //fout<<i+1<<" "<<ja[k]+1<<" "<<a[k]<<std::endl ;
	  //std::cout<<i+1<<" "<<ja[k]+1<<" "<<a[k]<<std::endl ;
        }
      }
      ia[n] = ia_s[n] ;

      matrixTransformTimer->stop();
      matrixTransformTimer->incrementNumCalls();

      // Upload matrix
      // TODO Do we need to pin memory here through AMGX_pin_memory?
      RCP<Teuchos::Time> matrixTimer = Teuchos::TimeMonitor::getNewTimer("MueLu: AMGX: transfer matrix  CPU->GPU");
      matrixTimer->start();
      if (numProcs == 1) 
      {
        std::cout<<"AMGX MATRIX UPLOAD FROM HOST"<<std::endl ;
        AMGX_matrix_upload_all(amgx_env.m_A,amgx_env.m_N,amgx_env.m_nnz, 1, 1, ia.data(), ja.data(), a.data(), NULL);
        std::cout<<"AMGX MATRIX UPLOAD SUCCESSFUL"<<std::endl ;


      } else {
        // Transform the matrix
        std::vector<int>    ia_new(ia.size());
        std::vector<int>    ja_new(ja.size());
        std::vector<double> a_new (a.size());

        ia_new[0] = 0;
        for (int i = 0; i <amgx_env.m_N ; i++) {
          int oldRow = amgx2muelu[i];

          ia_new[i+1] = ia_new[i] + (ia[oldRow+1] - ia[oldRow]);

          for (int j = ia[oldRow]; j < ia[oldRow+1]; j++) {
            int offset = j - ia[oldRow];
            ja_new[ia_new[i] + offset] = amgx_env.m_muelu2amgx[ja[j]];
            a_new [ia_new[i] + offset] = a[j];
          }
          // Do bubble sort on two arrays
          // NOTE: There are multiple possible optimizations here (even of bubble sort)
          bool swapped;
          do {
            swapped = false;

            for (int j = ia_new[i]; j < ia_new[i+1]-1; j++)
              if (ja_new[j] > ja_new[j+1]) {
                std::swap(ja_new[j], ja_new[j+1]);
                std::swap(a_new [j], a_new [j+1]);
                swapped = true;
              }
          } while (swapped == true);
        }

        AMGX_matrix_upload_all(amgx_env.m_A, amgx_env.m_N, amgx_env.m_nnz, 1, 1, &ia_new[0], &ja_new[0], &a_new[0], NULL);
      }

      /* set the connectivity information (for the vector) */
      AMGX_vector_bind(amgx_env.m_X, amgx_env.m_A);
      AMGX_vector_bind(amgx_env.m_Y, amgx_env.m_A);

      matrixTimer->stop();
      matrixTimer->incrementNumCalls();

      std::cout<<"AMGX SOLVER SETUP"<<std::endl ;
      AMGX_solver_setup(amgx_env.m_solver,amgx_env.m_A);
      std::cout<<"INIT AMGX ENV FROM MueLU MATRIX OK"<<std::endl ;

}
#endif
#endif

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
