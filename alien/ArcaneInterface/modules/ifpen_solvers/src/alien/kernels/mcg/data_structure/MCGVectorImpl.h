// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#include "mpi.h"

#include "alien/kernels/mcg/MCGBackEnd.h"
#include "alien/kernels/mcg/data_structure/MCGInternal.h"
#include "alien/kernels/mcg/data_structure/MCGVector.h"

namespace Alien {

template<typename NumT,MCGInternal::eMemoryDomain Domain>
MCGVector<NumT,Domain>::MCGVector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::mcgsolver>::name())
{}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
MCGVector<NumT,Domain>::~MCGVector()
{
  delete m_internal;
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
void
MCGVector<NumT,Domain>::init(const VectorDistribution& dist, const bool need_allocate)
{
  if (need_allocate)
    allocate();
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
void
MCGVector<NumT,Domain>::allocate()
{
  delete m_internal;

  const VectorDistribution& dist = this->distribution();
  int block_size = 1;

  if (this->block())
    block_size = this->block()->sizeX();
  else if (this->vblock())
    throw FatalErrorException(A_FUNCINFO, "Not implemented yet");

  m_internal = new VectorInternal(dist.localSize(), block_size);
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
void
MCGVector<NumT,Domain>::setValues(double const* values)
{
  // TODO: perform theses operations at object setup
  const VectorDistribution& dist = this->distribution();
  int block_size = 1;

  if (this->block())
    block_size = this->block()->sizeX();
  else if (this->vblock())
    throw FatalErrorException(A_FUNCINFO, "Not implemented yet for vblock");

  assert(block_size == m_internal->m_vector->blockSize());
  assert(dist.localSize() == m_internal->m_vector->size());

  double* data = m_internal->m_vector->data();
  for (int i = 0; i < dist.localSize() * block_size; ++i)
    data[i] = values[i];
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
void
MCGVector<NumT,Domain>::getValues(double* values) const
{
  // TODO: perform theses operations at object setup
  const VectorDistribution& dist = this->distribution();
  int block_size = 1;

  if (this->block())
    block_size = this->block()->sizeX();
  else if (this->vblock())
    throw FatalErrorException(A_FUNCINFO, "Not implemented yet for vblock");

  assert(block_size == m_internal->m_vector->blockSize());
  assert(dist.localSize() == m_internal->m_vector->size());

  const double* data = m_internal->m_vector->data();
  for (int i = 0; i < dist.localSize() * block_size; i++)
    values[i] = data[i];
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
void
MCGVector<NumT,Domain>::update(const MCGVector& v)
{
  ALIEN_ASSERT((this == &v), ("Unexpected error"));
}

} // namespace Alien
