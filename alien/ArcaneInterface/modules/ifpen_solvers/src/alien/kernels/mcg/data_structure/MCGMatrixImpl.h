// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0

#include "alien/core/impl/MultiMatrixImpl.h"

#include "alien/kernels/mcg/data_structure/MCGInternal.h"
#include "alien/kernels/mcg/data_structure/MCGMatrix.h"

#include "MCGSolver/LinearSystem/LinearSystem.h"
#include "MCGSolver/GPULinearSystem/GPULinearSystem.h"

namespace Alien {


template<typename NumT,MCGInternal::eMemoryDomain Domain>
MCGMatrix<NumT,Domain>::MCGMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, MCGInternal::AlgebraTraitsType<Domain>::name())
{
  m_internal = new MatrixInternal();
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
MCGMatrix<NumT,Domain>::~MCGMatrix()
{
  delete m_internal;
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
bool
MCGMatrix<NumT,Domain>::initMatrix(const MCGInternal::eMemoryDomain src_domain,const int block_size, const int block_size2,
    const int nrow,const int ncol,
    int const* row_offset, int const* cols, int partition_offset)
{
  Integer nblocks = row_offset[nrow];

  m_internal->m_elem_perm.resize(nblocks);

  if constexpr (Domain == MCGInternal::eMemoryDomain::Host) {
    if (src_domain == MCGInternal::eMemoryDomain::Host) {
      // CPU -> CPU
      m_internal->m_matrix =
        MCGSolver::LinearSystem<NumT,MCGSolver::Int32SparseIndex>::createMatrix(
          block_size, block_size2, nrow, ncol, nblocks, row_offset,cols,
          m_internal->m_elem_perm.data());
    }
    else {
      // GPU -> CPU
      throw Alien::FatalErrorException("Init CPU Matrix from GPU datas not implemented");
    }
  }
  else {
    if (src_domain == MCGInternal::eMemoryDomain::Host) {
      // CPU -> GPU
      m_internal->m_matrix = MCGSolver::GPULinearSystem<NumT,MCGSolver::Int32SparseIndex>::createMatrixFromHost(
        block_size, block_size2, nrow, ncol, nblocks, row_offset, cols,
        m_internal->m_elem_perm.data());

      m_internal->m_val.resize(nblocks*block_size*block_size2);
    }
    else {
      // GPU -> GPU
      m_internal->m_matrix =
        MCGSolver::GPULinearSystem<Real,MCGSolver::Int32SparseIndex>::createMatrix(
          block_size,block_size2,nrow,ncol,nblocks,row_offset,cols,m_internal->m_elem_perm.data());
    }
  }

  m_internal->m_elliptic_split_tag = computeEllipticSplitTags(block_size);
  m_is_init = true;
  return true;
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
bool
MCGMatrix<NumT,Domain>::initMatrixValues(const MCGInternal::eMemoryDomain src_domain,Real const* values)
{
  if constexpr (Domain == MCGInternal::eMemoryDomain::Host) {
    if (src_domain == MCGInternal::eMemoryDomain::Host) {
      // CPU -> CPU
      MCGSolver::LinearSystem<NumT,MCGSolver::Int32SparseIndex>::setMatrixValues(
           m_internal->m_matrix,values,m_internal->m_elem_perm.data());
    }
    else {
      // GPU -> CPU
      throw Alien::FatalErrorException("Init CPU Matrix values from GPU datas not implemented");
    }
  }
  else {
    if (src_domain == MCGInternal::eMemoryDomain::Host) {
      // CPU -> GPU
      auto error = cudaMemcpyAsync(m_internal->m_val.data(), values, m_internal->m_val.size()*sizeof(NumT),
        cudaMemcpyHostToDevice, nullptr);

      if (error != cudaSuccess) {
        throw Alien::FatalErrorException("Cuda Error");
      }

      MCGSolver::GPULinearSystem<NumT,MCGSolver::Int32SparseIndex>::setMatrixValues(m_internal->m_matrix,
        m_internal->m_val.data(), m_internal->m_elem_perm.data());
    }
    else {
      MCGSolver::GPULinearSystem<NumT,MCGSolver::Int32SparseIndex>::setMatrixValues(
          m_internal->m_matrix,values,m_internal->m_elem_perm.data());
    }
  }

  return true;
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
bool
MCGMatrix<NumT,Domain>::isInit() const
{
  return m_is_init;
}

template<typename NumT,MCGInternal::eMemoryDomain Domain>
bool
MCGMatrix<NumT,Domain>::computeEllipticSplitTags(int equation_num) const
{
  bool elliptic_split_tag_found = false;
  const ISpace& space = this->rowSpace();
  const MatrixDistribution& dist = this->distribution();
  Integer min_local_index = dist.rowOffset();
  Integer local_size = dist.localRowSize();

  m_internal->m_equation_type =
      std::make_shared<typename MatrixInternal::VectorEqType>(local_size * equation_num, 1);

  MCGSolver::ManagedArray<MCGSolver::Equation::eType> equation_type;

  MCGSolver::Equation::eType *eq_type = nullptr;

  if constexpr (Domain == MCGInternal::eMemoryDomain::Host) {
    eq_type = m_internal->m_equation_type->data();
  }
  else {
    equation_type.resize(local_size * equation_num);
    eq_type = equation_type.data();
  }

  for (int i = 0; i < local_size * equation_num; ++i)
    eq_type[i] = MCGSolver::Equation::NoType; // NoTyp == 0 , Elliptic==1 cf. PrecondEquation.h

  for (Integer i = 0; i < space.nbField(); ++i) {
    const UniqueArray<Integer>& indices = space.field(i);
    if (space.fieldLabel(i) == "Elliptic" and not indices.empty()) {
      elliptic_split_tag_found = true;
      for (Integer j = 0; j < indices.size(); ++j) {
        const Integer index = indices[j];
        eq_type[(index - min_local_index) * equation_num] = MCGSolver::Equation::Elliptic;
      }
    }
  }

  if (Domain == MCGInternal::eMemoryDomain::Device) {
    auto err = cudaMemcpyAsync(m_internal->m_equation_type->data(),eq_type,
      equation_type.size() * sizeof(MCGSolver::Equation::eType), cudaMemcpyHostToDevice, nullptr);

    if (err != cudaSuccess) {
     throw Alien::FatalErrorException("cudaMemcpyAsync for elliptic_equation tag failed");
    }
  }


  return elliptic_split_tag_found;
}
} // namespace Alien
