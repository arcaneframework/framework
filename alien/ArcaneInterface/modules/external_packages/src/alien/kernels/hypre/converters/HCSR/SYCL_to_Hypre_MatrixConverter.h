// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <alien/kernels/hypre/HypreBackEnd.h>
#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/data_structure/HypreInternal.h>

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/core/block/ComputeBlockOffsets.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>

#include <alien/kernels/hypre/data_structure/HypreMatrix.h>
#include <alien/kernels/hypre/HypreBackEnd.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SYCL_to_Hypre_MatrixConverter
: public IMatrixConverter
, public Alien::MatrixConverterT<Alien::BackEnd::tag::sycl,Alien::BackEnd::tag::hypre>
{
 public:
  using ConcreteBaseType = Alien::MatrixConverterT<Alien::BackEnd::tag::sycl,Alien::BackEnd::tag::hypre> ;

  SYCL_to_Hypre_MatrixConverter();
  virtual ~SYCL_to_Hypre_MatrixConverter() {}

 public:
  BackEndId sourceBackend() const { return AlgebraTraits<BackEnd::tag::sycl>::name(); }
  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::hypre>::name(); }
  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;
  void convert(const ConcreteBaseType::SourceMatrixType& source, ConcreteBaseType::TargetMatrixType& target) const;

  void _build(
      const SYCLBEllPackMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const;
  void _buildBlock(
      const SYCLBEllPackMatrix<Arccore::Real>& sourceImpl, HypreMatrix& targetImpl) const;
};

/*---------------------------------------------------------------------------*/
