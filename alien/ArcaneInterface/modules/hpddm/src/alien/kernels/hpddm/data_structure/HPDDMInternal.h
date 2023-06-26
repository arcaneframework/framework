/*
 * HPDDMInternal.h
 *
 *  Created on: Feb 6, 2020
 *      Author: gratienj
 */

#ifndef MODULES_ALIENIFPENSOLVERS_SRC_ALIEN_KERNELS_HPDDM_DATASTRUCTURE_HPDDMINTERNAL_H_
#define MODULES_ALIENIFPENSOLVERS_SRC_ALIEN_KERNELS_HPDDM_DATASTRUCTURE_HPDDMINTERNAL_H_

#ifdef ALIEN_USE_HPDDM
#define HPDDM_NUMBERING 'C'
#define DMUMPS 1
#define MUMPSSUB 1
#define MU_ARPACK 1

#include <HPDDM.hpp>
#endif

#include <vector>
#include <list>

BEGIN_HPDDMINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
struct HPDDMInternal
{
  static void initialize(Arccore::MessagePassing::IMessagePassingMng* parallel_mng);
  static void finalize();

  template <typename ValueT>
  static ValueT getEnv(std::string const& key, ValueT default_value);

 private:
  static bool m_is_initialized;
};

template <typename ValueT> class MatrixInternal
{
 public:
  typedef SimpleCSRMatrix<ValueT> CSRMatrixType;
  typedef SimpleCSRVector<ValueT> CSRVectorType;

  int getNDofs() const { return m_ndofs; }

  int getNnz() const { return m_nnz; }

#ifdef ALIEN_USE_HPDDM
  typedef HPDDM::MatrixCSR<ValueT> HPDDMCSRMatrixType;
  typedef HPDDM::underlying_type<ValueT> HPDDMValueType;
  typedef HPDDM::Schwarz<SUBDOMAIN, COARSEOPERATOR, 'S', ValueT> HPDDMMatrixType;

  void compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& A,
      unsigned short nu, bool schwarz_coarse_correction);

  void compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& Ad,
      const CSRMatrixType& An, unsigned short nu, bool schwarz_coarse_correction);

  HPDDMMatrixType& matrix() { return m_matrix; }
#endif
 private:
  void _computeMPIGhostMatrix(const CSRMatrixType& A);
  HPDDMCSRMatrixType* _createDirchletMatrix(const CSRMatrixType& A);
  HPDDMCSRMatrixType* _createNeunmanMatrix(const CSRMatrixType& A);

  void _computeUnitPartition();

  void _computeOverlapConnectivity(const CSRMatrixType& A);

  void _compute(HPDDMCSRMatrixType* matrix_dirichlet, HPDDMCSRMatrixType* matrix_neumann,
      unsigned short nu, bool schwarz_coarse_correction);

  IMessagePassingMng* m_parallel_mng = nullptr;

  int m_ndofs = 0;
  int m_nnz = 0;

  int m_local_nrows = 0;
  int m_local_nnz = 0;
  int m_ghost_nrows = 0;
  int m_ghost_nnz = 0;
  std::vector<ValueT> m_diag_correction;

  CSRMatrixType m_mpi_ext_int_matrix;
  CSRMatrixType m_mpi_ext_ext_matrix;
  CSRMatrixType m_ghost_matrix_dirichlet;
  CSRMatrixType m_ghost_matrix_neumann;
  CSRMatrixType m_csr_matrix_dirichlet;
  CSRMatrixType m_csr_matrix_neumann;

#ifdef ALIEN_USE_HPDDM
  std::list<int> m_overlap;
  std::vector<std::vector<int>> m_mapping;
  std::vector<HPDDMValueType> m_unit_partition;

  HPDDMMatrixType m_matrix;
#endif
};

/*---------------------------------------------------------------------------*/

template <typename ValueT> class VectorInternal
{
 public:
  std::vector<ValueT> m_internal;
};
/*---------------------------------------------------------------------------*/

END_HPDDMINTERNAL_NAMESPACE

#endif /* MODULES_ALIENIFPENSOLVERS_SRC_ALIEN_KERNELS_HPDDM_DATASTRUCTURE_HPDDMINTERNAL_H_ \
          */
