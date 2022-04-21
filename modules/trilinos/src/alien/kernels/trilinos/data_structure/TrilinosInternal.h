#ifndef ALIEN_TRILINOSIMPL_TRILINOSINTERNAL_H
#define ALIEN_TRILINOSIMPL_TRILINOSINTERNAL_H
/* Author :
 */

//! Internal struct for TRILINOS implementation
/*! Separate data from header;
 *  can be only included by LinearSystem and LinearSolver
 */

#include <string>
#include <alien/distribution/MatrixDistribution.h>

#include <alien/kernels/trilinos/TrilinosPrecomp.h>

#ifdef ALIEN_USE_TRILINOS
#ifdef ALIEN_USE_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif

#include <Teuchos_DefaultMpiComm.hpp> // wrapper for MPI_Comm
#include <Tpetra_Version.hpp> // Tpetra version string#endif

#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>

#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <KokkosCompat_ClassicNodeAPI_Wrapper.hpp>

#include <MatrixMarket_Tpetra.hpp>

#define HAVE_MUELU
#ifdef HAVE_MUELU
#include <Trilinos_version.h>
#include <MueLu.hpp>

#if defined(HAVE_MUELU_AMGX)
#include <MueLu_AMGXOperator.hpp>
//#include <MueLu_AMGX_Setup.hpp>
#endif // HAVE_MUELU_AMGX
#endif // HAVE_MUELU

#endif // ALIEN_USE_TRILINOS

/*---------------------------------------------------------------------------*/

BEGIN_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

//! Check parallel feature for MTL
struct Features
{
  static void checkParallel(const MatrixDistribution& dist) {}
};

/*---------------------------------------------------------------------------*/
#ifdef HAVE_MUELU_AMGX
struct AMGXEnv
{
   AMGX_Mode               m_mode ;
   AMGX_config_handle      m_config    = nullptr ;
   AMGX_resources_handle   m_resources = nullptr ;
   AMGX_solver_handle      m_solver    = nullptr ;
   AMGX_matrix_handle      m_A         = nullptr ;
   AMGX_vector_handle      m_X         = nullptr ;
   AMGX_vector_handle      m_Y         = nullptr ;
   int                     m_N         = 0 ;
   int                     m_nnz       = 0 ;

   std::vector<int>        m_muelu2amgx;

   std::string             m_config_file ;
   std::string             m_config_string ;
} ;
void solveAMGX(AMGXEnv& amgx_env,double* x, double* y) ;
#endif
struct TrilinosInternal
{
  template <typename tpetra_tag> struct Node;

  static void initialize(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      std::string const& execution_space = "Undefined", int nb_threads = 0,
      bool use_amgx = false);

  static void finalize();

  static void initMPIEnv(MPI_Comm comm);

  template <typename ValueT>
  static ValueT getEnv(std::string const& key, ValueT default_value);

  static int const& getNbThreads() { return m_nb_threads; }

  static std::string const& getExecutionSpace() { return m_execution_space; }

 public :
  static bool        m_amgx_is_initialized ;
 private:
  static bool        m_is_initialized;
  static int         m_nb_threads;
  static std::size_t m_nb_hyper_threads;
  static std::size_t m_mpi_core_id_offset;
  static std::string m_execution_space;

  static std::unique_ptr<const Teuchos::Comm<int>> m_trilinos_comm;
};


#ifdef HAVE_MUELU_AMGX
void initAMGX(AMGXEnv& amgx_env) ;
#endif
#ifdef KOKKOS_ENABLE_SERIAL
template <> struct TrilinosInternal::Node<BackEnd::tag::tpetraserial>
{
  typedef Kokkos::Compat::KokkosSerialWrapperNode type;
  static const std::string name;
  static const std::string execution_space_name;
};

#ifdef HAVE_MUELU_AMGX
inline void initAMGX(AMGXEnv& amgx_env,
		     Tpetra::CrsMatrix<double, int, int,TrilinosInternal::Node<BackEnd::tag::tpetraserial>::type> const& matrix)
{
}
#endif
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <> struct TrilinosInternal::Node<BackEnd::tag::tpetraomp>
{
  typedef Kokkos::Compat::KokkosOpenMPWrapperNode type;
  static const std::string name;
  static const std::string execution_space_name;
};
#ifdef HAVE_MUELU_AMGX
inline void initAMGX(AMGXEnv& amgx_env,
		     Tpetra::CrsMatrix<double, int, int,TrilinosInternal::Node<BackEnd::tag::tpetraomp>::type> const& matrix)
{
}
#endif
#endif

#ifdef KOKKOS_ENABLE_THREADS
template <> struct TrilinosInternal::Node<BackEnd::tag::tpetrapth>
{
  typedef Kokkos::Compat::KokkosThreadsWrapperNode type;
  static const std::string name;
  static const std::string execution_space_name;
};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <> struct TrilinosInternal::Node<BackEnd::tag::tpetracuda>
{
  typedef Kokkos::Compat::KokkosCudaWrapperNode type;
  static const std::string name;
  static const std::string execution_space_name;
};
#ifdef HAVE_MUELU_AMGX
void initAMGX(AMGXEnv& amgx_env,
	      Tpetra::CrsMatrix<double, int, int,TrilinosInternal::Node<BackEnd::tag::tpetracuda>::type> const& matrix) ;
#endif
#endif

#ifdef KOKKOS_ENABLE_SERIAL
template <typename ValueT, typename TagT = BackEnd::tag::tpetraserial>
#else
#ifdef KOKKOS_ENABLE_OPENMP
template <typename ValueT, typename TagT = BackEnd::tag::tpetraomp>
#endif
#endif
class MatrixInternal
{
 public:
  typedef ValueT scalar_type;
  typedef typename TrilinosInternal::Node<TagT>::type node_type;
  typedef Tpetra::Map<int, int, node_type> map_type;
  typedef Tpetra::CrsGraph<int, int, node_type> graph_type;
  typedef typename graph_type::local_ordinal_type local_ordinal_type;
  typedef typename graph_type::global_ordinal_type global_ordinal_type;

  typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>
      matrix_type;

  typedef Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>
      vector_type;
  
  typedef Teuchos::ScalarTraits<scalar_type>                 STS;
  typedef typename STS::magnitudeType                        magnitude_type;
  typedef typename STS::coordinateType                       real_type;

  typedef Tpetra::MultiVector<real_type,
                              local_ordinal_type,
                              global_ordinal_type,
                              node_type>                     coord_vector_type;
  
  MatrixInternal(int local_offset, int global_size, int local_size, MPI_Comm const& comm)
  {
    using Teuchos::Array;
    using Teuchos::ArrayView;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::Comm;
    using Teuchos::MpiComm;

    Array<global_ordinal_type> indices(local_size);
    for (int i = 0; i < local_size; ++i)
      indices[i] = local_offset + i;
    const Tpetra::global_size_t gsize = global_size;
    m_comm = rcp(new MpiComm<int>(comm));
    m_map = rcp(new map_type(gsize, indices, 0, m_comm));

    //m_internal.reset(new matrix_type(this->m_map, 0, Tpetra::ProfileType(1)));
    m_internal.reset(new matrix_type(this->m_map, 10));
  }

  bool initMatrix(int local_offset, int nrows, int const* kcol, int const* cols,
      int block_size, ValueT const* values);

  bool setMatrixValues(Real const* values);

  void mult(vector_type const& x, vector_type& y) const;
  void mult(ValueT const* x, ValueT* y) const;

  int m_block_size = 1;
  bool m_is_parallel = false;
  int m_local_offset = 0;
  int m_local_size = 0;

  Teuchos::RCP<const Teuchos::Comm<int>> m_comm;
  Teuchos::RCP<const map_type>           m_map;
  Teuchos::RCP<coord_vector_type>        m_coordinates;

  std::unique_ptr<matrix_type> m_internal;
};

/*---------------------------------------------------------------------------*/

#ifdef KOKKOS_ENABLE_SERIAL
template <typename ValueT, typename TagT = BackEnd::tag::tpetraserial>
#else
#ifdef KOKKOS_ENABLE_OPENMP
template <typename ValueT, typename TagT = BackEnd::tag::tpetraomp>
#endif
#endif
class VectorInternal
{
 public:
  typedef ValueT scalar_type;
  typedef typename TrilinosInternal::Node<TagT>::type node_type;
  typedef Tpetra::Map<int, int, node_type> map_type;
  typedef Tpetra::CrsGraph<int, int, node_type> graph_type;
  typedef typename graph_type::local_ordinal_type local_ordinal_type;
  typedef typename graph_type::global_ordinal_type global_ordinal_type;

  typedef Tpetra::Vector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>
      vector_type;

  typedef Tpetra::CrsMatrix<scalar_type, local_ordinal_type, global_ordinal_type,
      node_type>
      matrix_type;

  VectorInternal(std::size_t local_offset, std::size_t global_size,
      std::size_t local_size, MPI_Comm const& comm)
  {
    using Teuchos::Array;
    using Teuchos::ArrayView;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::Comm;
    using Teuchos::MpiComm;

    Array<global_ordinal_type> indices(local_size);
    for (int i = 0; i < local_size; ++i)
      indices[i] = local_offset + i;
    m_comm = rcp(new MpiComm<int>(comm));
    m_map = rcp(new map_type(global_size, indices(), 0, m_comm));

    m_internal.reset(new vector_type(this->m_map));
  }
  Teuchos::RCP<const Teuchos::Comm<int>> m_comm;
  Teuchos::RCP<const map_type> m_map;

  std::unique_ptr<vector_type> m_internal;
};
/*---------------------------------------------------------------------------*/

END_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/
#endif /* ALIEN_MTLIMPL_MTLINTERNAL_H */
