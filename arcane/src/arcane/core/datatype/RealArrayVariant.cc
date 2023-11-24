// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealArrayVariant.cc                                         (C) 2000-2023 */
/*                                                                           */
/* Variant pouvant contenir les types ConstArrayView, Real2 et Real3.        */
/*---------------------------------------------------------------------------*/

#include "arcane/datatype/RealArrayVariant.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NumVector.h"

#include "arcane/MathUtils.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT
void _arcaneTestRealArrayVariant()
{
  UniqueArray<Real> a1_({2.4, 5.6, 3.3, 5.4});
  ConstArrayView<Real> a1 = a1_.constView();
  Real2 a2{ 2.0, 3.1 };
  Real3 a3{ 4.0, 7.2, 3.6 };

  const Integer nb_variants = 3;
  RealArrayVariant variants[nb_variants] = { RealArrayVariant(a1), RealArrayVariant(a2), RealArrayVariant(a3) };

  for (Integer v=0 ; v<nb_variants ; ++v){
    std::cout << "A" << v << "=[ ";
    for (Integer i=0 ; i<variants[v].size() ; ++i)
      std::cout << variants[v][i] << " ";
    std::cout << "]\n";
  }

#if defined(ARCANE_HAS_ACCELERATOR_API)
  NumArray<Real,MDDim1> num_data(4, { 2.4, 5.6, 3.3, 5.4 });
  RealArrayVariant variant2{num_data};
  NumArray<Real,MDDim1> num_data_copy(variant2);
  if (num_data_copy.to1DSpan()!=num_data.to1DSpan())
    ARCANE_FATAL("Bad value for copy");
#endif

  RealN2 b2{ 2.0, 3.1 };
  RealN3 b3{ 4.0, 7.2, 3.6 };
  NumVector<Real,3> b4{ 2.0, 1.2, 4.6 };
  RealArrayVariant b2_variant(b2);
  RealArrayVariant b3_variant(b3);
  NumVector<Real,2> c2(b2_variant);
  NumVector<Real,3> c3(b3_variant);
  auto z = c3 + b4;
  std::cout << "Z=" << z.vx() << "\n";
  std::cout << "NORM=" << math::normalizedCrossProduct3(c3,b4).normL2();
  for (Integer i=0 ; i<3 ; ++i)
    std::cout << "V=" << i << " v=" << z(i) << "\n"; 
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
