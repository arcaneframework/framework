// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosInternal.h>

#include <alien/core/block/Block.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
#include "TrilinosVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT>
TrilinosVector<ValueT, TagT>::TrilinosVector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<TagT>::name())
, m_local_offset(0)
{
}

/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT> TrilinosVector<ValueT, TagT>::~TrilinosVector()
{
}

/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT>
void
TrilinosVector<ValueT, TagT>::init(const VectorDistribution& dist, const bool need_allocate)
{
  const Block* block = this->block();
  if (this->block())
    m_block_size *= block->size();
  else if (this->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");
  else
    m_block_size = 1;
  if (need_allocate)
    allocate();
}

/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT>
void
TrilinosVector<ValueT, TagT>::allocate()
{
  const VectorDistribution& dist = this->distribution();
  m_local_offset = dist.offset() * m_block_size;
  m_global_size  = dist.globalSize() * m_block_size;
  m_local_size   = dist.localSize() * m_block_size;
  auto* parallel_mng =
      const_cast<Arccore::MessagePassing::IMessagePassingMng*>(dist.parallelMng());

  using namespace Arccore::MessagePassing::Mpi;
  auto* pm = dynamic_cast<MpiMessagePassingMng*>(parallel_mng);
  if(pm && *static_cast<const MPI_Comm*>(pm->getMPIComm()) != MPI_COMM_NULL)
    m_internal.reset(
        new VectorInternal(dist.offset(), m_global_size, m_local_size, *static_cast<const MPI_Comm*>(pm->getMPIComm())));
  else
    m_internal.reset(
        new VectorInternal(m_local_offset, m_global_size, m_local_size, MPI_COMM_WORLD));
  // m_internal->m_internal = 0.;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT, typename TagT>
void
TrilinosVector<ValueT, TagT>::setValues(const int nrow, const ValueT* values)
{
  auto& x = *m_internal->m_internal;
#if (TRILINOS_MAJOR_VERSION < 15)
  x.sync_host();
  auto x_2d = x.getLocalViewHost();
  auto x_1d = Kokkos::subview(x_2d, Kokkos::ALL(), 0);
  x.modify_host();
#else
  auto x_2d = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  auto x_1d = Kokkos::subview(x_2d, Kokkos::ALL(), 0);
#endif

  for (int i = 0; i < nrow; ++i) {
    x_1d(i) = values[i];
    // std::cout<<"SET X["<<i<<"]"<<values[i]<<std::endl ;
  }
#if (TRILINOS_MAJOR_VERSION < 15)
  using memory_space = typename VectorInternal::vector_type::device_type::memory_space;
  x.template sync<memory_space>();
#endif
}

/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT>
void
TrilinosVector<ValueT, TagT>::getValues(const int nrow, ValueT* values) const
{
  auto& x = *m_internal->m_internal;
#if (TRILINOS_MAJOR_VERSION < 15)
  x.sync_host();
  auto x_2d = x.getLocalViewHost();
#else
  auto x_2d = x.getLocalViewHost(Tpetra::Access::ReadWrite);
#endif
  auto x_1d = Kokkos::subview(x_2d, Kokkos::ALL(), 0);
  for (int i = 0; i < nrow; ++i) {
    values[i] = x_1d(i);
    // std::cout<<"GET X["<<i<<"]"<<values[i]<<std::endl ;
  }
}

template <typename ValueT, typename TagT>
ValueT
TrilinosVector<ValueT, TagT>::norm1() const
{
  return m_internal->m_internal->norm1();
}

template <typename ValueT, typename TagT>
ValueT
TrilinosVector<ValueT, TagT>::norm2() const
{
  return m_internal->m_internal->norm2();
}

template <typename ValueT, typename TagT>
ValueT
TrilinosVector<ValueT, TagT>::dot(TrilinosVector const& y) const
{
  return m_internal->m_internal->dot(*y.m_internal->m_internal);
}

template <typename ValueT, typename TagT>
void
TrilinosVector<ValueT, TagT>::dump() const
{
  auto& x = *m_internal->m_internal;
#if (TRILINOS_MAJOR_VERSION < 15)
  x.sync_host();
  auto x_2d = x.getLocalViewHost();
#else
  auto x_2d = x.getLocalViewHost(Tpetra::Access::ReadOnly);
#endif
  auto x_1d = Kokkos::subview(x_2d, Kokkos::ALL(), 0);
  const size_t localLength = x.getLocalLength();
  for (int i = 0; i < localLength; ++i)
    std::cout << x_1d(i) << " " << std::endl;
}

template <typename ValueT, typename TagT>
void
TrilinosVector<ValueT, TagT>::dump(std::string const& filename) const
{
  Tpetra::MatrixMarket::Writer<typename VectorInternal::matrix_type>::writeDenseFile(
      filename, *m_internal->m_internal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifdef KOKKOS_ENABLE_SERIAL
template class TrilinosVector<double, BackEnd::tag::tpetraserial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template class TrilinosVector<double, BackEnd::tag::tpetraomp>;
#endif
#ifdef KOKKOS_ENABLE_THREADS
template class TrilinosVector<double, BackEnd::tag::tpetrapth>;
#endif
#ifdef KOKKOS_ENABLE_CUDA
template class TrilinosVector<double, BackEnd::tag::tpetracuda>;
#endif

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
