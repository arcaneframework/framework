// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealArray2Variant.cc                                        (C) 2000-2023 */
/*                                                                           */
/* Variant pouvant contenir les types ConstArray2View, Real2x2 et Real3x3.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/datatype/RealArray2Variant.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NumMatrix.h"

#include "arcane/MathUtils.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void
_arcaneTestRealArray2Variant()
{
  // NOTE: Les dimensions max pour un RealArray2Variant sont 3x3.
  Real data[4] = {2.4, 5.6,3.3, 4.4};
  ConstArray2View<Real> a(data, 2, 2);
  Real2x2 a22{ Real2{ -1.0, -2.5 }, Real2{ -2.0, 3.7 } };
  Real3x3 a33{ Real3{ -2.1, 3.9, 1.5 }, Real3{ 9.2, 3.4, 2.1 }, Real3{ 7.1, 4.5, 3.2 } };

  const Integer nb_variants = 3;
  RealArray2Variant variants[nb_variants] = { RealArray2Variant(a), RealArray2Variant(a22), RealArray2Variant(a33) };

  for (Integer v=0 ; v<nb_variants ; ++v) {
    std::cout << "A" << v << "=[ ";
    for (Integer i=0 ; i<variants[v].dim1Size() ; ++i) {
      std::cout << "[ ";
      for (Integer j=0 ; j<variants[v].dim2Size() ; ++j)
        std::cout << variants[v][i][j] << " ";
      std::cout << "]\n";
    }
    std::cout << "]\n";
  }

#if defined(ARCANE_HAS_ACCELERATOR_API)
  NumArray<Real,MDDim2> num_data(3, 2, {1.4, 2.3, 4.5, 5.7, 2.9 , 6.5 });
  RealArray2Variant variant2{num_data};
  NumArray<Real,MDDim2> num_data_copy(variant2);
  Span<const Real> variant2_span(variant2.data(),variant2.dim1Size()*variant2.dim2Size());
  std::cout << "NUM_DATA     =" << num_data.to1DSpan() << "\n";
  std::cout << "NUM_DATA_COPY=" << num_data_copy.to1DSpan() << "\n";
  if (num_data_copy.to1DSpan()!=num_data.to1DSpan())
    ARCANE_FATAL("Bad value for copy");
#endif

  RealN2x2 n22(a22);
  RealN3x3 n33(a33);
  NumMatrix<Real,3> m33(Real3x3( Real3{ 1.1, 2.9, 3.5 }, Real3{ 2.2, 4.4, 6.1 }, Real3{ -1.1, 2.5, 1.2 } ));
  RealArray2Variant n22_variant(n22);
  RealArray2Variant n33_variant(n33);
  RealN2x2 c22(n22_variant);
  RealN3x3 c33(n33_variant);

  auto z = m33 + c33;
  std::cout << "Z=" << z.vx().vx() << "\n";
  std::cout << "NORM=" << math::matrixDeterminant(math::matrix3x3Prod(z,c33));
  for (Integer i=0 ; i<3 ; ++i)
    for (Integer j=0 ; j<3 ; ++j)
      std::cout << "V1=" << i << " "  << j << " v=" << m33(i,j) << "\n"; 
  for (Integer i=0 ; i<3 ; ++i){
    auto lc = m33(i);
    for (Integer j=0 ; j<3 ; ++j)
      std::cout << "V2=" << i << " "  << j << " v=" << lc(j) << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
