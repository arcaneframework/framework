// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

//! Internal struct for HTS implementation
/*! Separate data from header;
 *  can be only included by LinearSystem and LinearSolver
 */

#include <alien/distribution/MatrixDistribution.h>

#ifdef ALIEN_USE_HARTS
#include "HARTS/HARTS.h"
#endif

#ifdef ALIEN_USE_HTSSOLVER
#include "HARTSSolver/HARTSSolverExport.h"
#include "HARTSSolver/HTSSolverConfig.h"
#include "HARTSSolver/HTS.h"
#include "HARTSSolver/MatrixVector/CSR/CSRProfile.h"
#include "HARTSSolver/MatrixVector/CSR/CSRMatrix.h"

#include "HARTSSolver/Utils/TraceMng.h"
#include "HARTSSolver/Utils/ArrayUtils.h"
#include "HARTSSolver/Utils/IParallelMng.h"
#include "HARTSSolver/Utils/BaseException.h"
#include "HARTSSolver/Utils/MPI/MPIParallelMng.h"
#include "HARTSSolver/Utils/MPI/MPIEnv.h"

#include "HARTSSolver/MatrixVector/ProfileInfo.h"
#include "HARTSSolver/MatrixVector/Allocator.h"
#include "HARTSSolver/Graph/MatrixGraph.h"
#include "HARTSSolver/Graph/SimpleGraph.h"
#include "HARTSSolver/Graph/GraphColorPartitioner.h"
#include "HARTSSolver/Graph/Partition.h"
#include "HARTSSolver/Graph/MPIPartition.h"
#include "HARTSSolver/Graph/PartitionFactory.h"
#include "HARTSSolver/MatrixVector/CSR/CSRMatrix.h"
#if defined(USE_AVX512)
#include "HARTSSolver/MatrixVector/CSR/SellCSMatrix.h"
#include "HARTSSolver/MatrixVector/CSR/MatrixProxyT.h"
#endif
#include "HARTSSolver/MatrixVector/Algebra.h"
#include "HARTSSolver/MatrixVector/DistUtils/SubGraphDomain.h"
#include "HARTSSolver/MatrixVector/DistUtils/SendRecvOp.h"
#include "HARTSSolver/MatrixVector/DistUtils/DistStructInfo.h"
#include "HARTSSolver/MatrixVector/MCCSR/MCCSRMatrix.h"
#endif

/*---------------------------------------------------------------------------*/

BEGIN_HTSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

//! Check parallel feature for MTL
struct Features
{
  static void checkParallel(const MatrixDistribution& dist) {}
};

/*---------------------------------------------------------------------------*/
struct HTSInternal
{
  typedef HartsSolver::MPIInfo MPIEnvType;
  typedef RunTimeSystem::ThreadEnv ThreadEnvType;
  typedef RunTimeSystem::MachineInfo MachineInfoType;
  typedef RunTimeSystem::ThreadSystemTopology ThreadSystemTopologyType;

  static void initialize(Arccore::MessagePassing::IMessagePassingMng* parallel_mng);
  static void finalize();
  static MPIEnvType* getMPIEnv() { return m_mpi_env.get(); }

  static void initMPIEnv(MPI_Comm comm);

  template <typename ValueT>
  static ValueT getEnv(std::string const& key, ValueT default_value);

 private:
  static bool m_is_initialized;
  static int m_nb_threads;
  static std::size_t m_nb_hyper_threads;
  static std::size_t m_mpi_core_id_offset;
  static std::unique_ptr<MachineInfoType> m_machine_info;
  static std::unique_ptr<ThreadSystemTopologyType> m_topology;

  static std::unique_ptr<ThreadEnvType> m_thread_env;
  static std::unique_ptr<MPIEnvType> m_mpi_env;
};

template <typename ValueT, bool is_mpi = true> class MatrixInternal
{
 public:
#ifdef USE_NUMA_ALLOCATOR
  typedef HartsSolver::MatrixVector::NumaAllocator AllocatorType;
#else
#ifdef USE_HBW_ALLOCATOR
  typedef HartsSolver::MatrixVector::HBWAllocator AllocatorType;
#else
  typedef HartsSolver::MatrixVector::DefaultAllocator AllocatorType;
#endif
#endif

  typedef HartsSolver::CSRMatrix<ValueT, 1, AllocatorType> CpuMatrixType;
#ifdef SIMD_VERSION
#if defined(USE_AVX512)
  typedef HartsSolver::MatrixVector::SellCSMatrix<ValueT, AllocatorType> SimdMatrixType;
  typedef HartsSolver::MatrixVector::CSRMatrixProxyT<CpuMatrixType, SimdMatrixType>
      MatrixType;
#else
  typedef CpuMatrixType MatrixType;
#endif

#else
  typedef CpuMatrixType MatrixType;
#endif

  typedef MatrixType MCMatrixType;
  typedef HartsSolver::DistStructInfo DistStructInfoType;

  typedef typename MatrixType::ProfileType ProfileType;
  typedef typename ProfileType::InfoVectorType InfoVectorType;
  typedef typename MatrixType::VectorType VectorType;
  typedef typename MCMatrixType::VectorDataType VectorDataType;

  typedef Graph::Partition GraphPartitionType;
  typedef RunTimeSystem::BasePartitioner<Graph::Partition> PartitionerType;
  typedef Graph::MPIPartition MPIPartitionType;

  typedef HartsSolver::MCCSRMatrix<MatrixType, Graph::Partition, is_mpi> MCCSRMatrixType;
  typedef typename MCCSRMatrixType::DDProfileType DDProfileType;

  typedef HartsSolver::CSRProfile MCProfileType;
  typedef HartsSolver::ProfileView MCProfileViewType;
  typedef MCProfileType::PermutationType MCProfilePermType;

  MatrixInternal() {}

  bool initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng, int nrows,
      int const* kcol, int const* cols, int block_size);

  bool setMatrixValues(Arccore::Real const* values);

  bool computeDDMatrix();

  void mult(ValueT const* x, ValueT* y);

  int m_block_size = 1;
  bool m_is_parallel = false;

  std::unique_ptr<MPIPartitionType> m_partition_info;
  std::unique_ptr<Graph::Partition> m_partition;
  std::unique_ptr<PartitionerType> m_rs_partition;
  std::unique_ptr<MCProfileType> m_profile;
  std::unique_ptr<MCProfilePermType> m_profile_permutation;
  std::unique_ptr<MCMatrixType> m_matrix;
  std::unique_ptr<DistStructInfoType> m_dist_info;

 public:
  std::unique_ptr<DDProfileType> m_dd_profile;
  std::unique_ptr<MCCSRMatrixType> m_dd_matrix;
};

/*---------------------------------------------------------------------------*/

template <typename ValueT, bool is_mpi = true> class VectorInternal
{
 public:
#ifdef USE_NUMA_ALLOCATOR
  typedef HartsSolver::MatrixVector::NumaAllocator AllocatorType;
#else
#ifdef USE_HBW_ALLOCATOR
  typedef HartsSolver::MatrixVector::HBWAllocator AllocatorType;
#else
  typedef HartsSolver::MatrixVector::DefaultAllocator AllocatorType;
#endif
#endif

  typedef HartsSolver::CSRMatrix<ValueT, 1, AllocatorType> CpuMatrixType;
#ifdef SIMD_VERSION
#if defined(USE_AVX512)
  typedef HartsSolver::MatrixVector::SellCSMatrix<ValueT, AllocatorType> SimdMatrixType;
  typedef HartsSolver::MatrixVector::CSRMatrixProxyT<CpuMatrixType, SimdMatrixType>
      MatrixType;
#else
  typedef CpuMatrixType MatrixType;
#endif

#else
  typedef CpuMatrixType MatrixType;
#endif

  typedef MatrixType MCMatrixType;
  typedef HartsSolver::DistStructInfo DistStructInfoType;

  typedef typename MatrixType::ProfileType ProfileType;
  typedef typename ProfileType::InfoVectorType InfoVectorType;
  typedef typename MatrixType::VectorType VectorType;
  typedef typename MCMatrixType::VectorDataType VectorDataType;

  VectorInternal(std::size_t local_size)
  : m_local_size(local_size)
  {
  }

  // VectorType m_data ;
  VectorDataType const* m_data = nullptr;
  std::size_t m_local_size = 0;
};
/*---------------------------------------------------------------------------*/

END_HTSINTERNAL_NAMESPACE

