// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealArray2Variant.cc                                        (C) 2000-2022 */
/*                                                                           */
/* Variant pouvant contenir les types ConstArray2View, Real2x2 et Real3x3.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/datatype/RealArray2Variant.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT void _arcaneTestRealArray2Variant()
{
  Real data[4] = {2.4, 5.6,3.3, 4.4};
  ConstArray2View<Real> a(data, 2, 2);
  Real2x2 a22{ Real2{ -1.0, -2.5 }, Real2{ -2.0, 3.7 } };
  Real3x3 a33{ Real3{ -2.1, 3.9, 1.5 }, Real3{ 9.2, 3.4, 2.1 }, Real3{ 7.1, 4.5, 3.2 } };

  NumArray<Real,2> num_data(3,3);

  const Integer nb_variants = 3;
  RealArray2Variant variants[nb_variants] = { RealArray2Variant(a), RealArray2Variant(a22), RealArray2Variant(a33) };

  for (Integer v=0 ; v<nb_variants ; ++v)
  {
    std::cout << "A" << v << "=[ ";
    for (Integer i=0 ; i<variants[v].dim1Size() ; ++i)
    {
      std::cout << "[ ";
      for (Integer j=0 ; j<variants[v].dim2Size() ; ++i)
        std::cout << variants[v][i][j] << " ";
      std::cout << "]\n";
    }
    std::cout << "]\n";
  }

  RealArray2Variant variant2{num_data};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
