// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/core/backend/MatrixConverterRegisterer.h>

#include <iostream>

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

#include <alien/kernels/composyx/ComposyxBackEnd.h>

#include <alien/kernels/composyx/data_structure/ComposyxMatrix.h>

using namespace Alien;
using namespace Alien::SimpleCSRInternal;

/*---------------------------------------------------------------------------*/

class SimpleCSR_to_Composyx_MatrixConverter : public IMatrixConverter
{
 public:
  SimpleCSR_to_Composyx_MatrixConverter() {}
  virtual ~SimpleCSR_to_Composyx_MatrixConverter() {}
 public:
  BackEndId sourceBackend() const
  {
    return AlgebraTraits<BackEnd::tag::simplecsr>::name();
  }

  BackEndId targetBackend() const { return AlgebraTraits<BackEnd::tag::composyx>::name(); }

  void convert(const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const;

  void _build(const SimpleCSRMatrix<Real>& sourceImpl,
      ComposyxMatrix<Real>& targetImpl) const;

  void _buildBlock(const SimpleCSRMatrix<Real>& sourceImpl,
      ComposyxMatrix<Real>& targetImpl) const;
};

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

void
SimpleCSR_to_Composyx_MatrixConverter::convert(
    const IMatrixImpl* sourceImpl, IMatrixImpl* targetImpl) const
{
  const SimpleCSRMatrix<Real>& v =
      cast<SimpleCSRMatrix<Real>>(sourceImpl, sourceBackend());
  ComposyxMatrix<Real>& v2 =
      cast<ComposyxMatrix<Real>>(targetImpl, targetBackend());

  alien_debug([&] {
    cout() << "Converting SimpleCSRMatrix: " << &v << " to ComposyxMatrix " << &v2;
  });

  if (targetImpl->block())
    _buildBlock(v, v2);
  else if (targetImpl->vblock())
    throw FatalErrorException(
        A_FUNCINFO, "Block sizes are variable - builds not yet implemented");
  else
    _build(v, v2);
}

void
SimpleCSR_to_Composyx_MatrixConverter::_build(
    const SimpleCSRMatrix<Real>& sourceImpl, ComposyxMatrix<Real>& targetImpl) const
{
  typedef SimpleCSRMatrix<Real>::MatrixInternal CSRMatrixType;
  const MatrixDistribution& dist = targetImpl.distribution();
  if(not targetImpl.compute(dist.parallelMng(),sourceImpl)){
      throw FatalErrorException(A_FUNCINFO, "Composyx Initialisation failed");
  }
}

void
SimpleCSR_to_Composyx_MatrixConverter::_buildBlock(
    const SimpleCSRMatrix<Real>& sourceImpl, ComposyxMatrix<Real>& targetImpl) const
{
}


REGISTER_MATRIX_CONVERTER(SimpleCSR_to_Composyx_MatrixConverter);
