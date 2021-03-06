// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RealArrayVariant.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Variant pouvant contenir les types ConstArrayView, Real2 et Real3.        */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/datatype/RealArrayVariant.h"

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
  NumArray<Real,1> num_data(4, { 2.4, 5.6, 3.3, 5.4 });

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

  RealArrayVariant variant2{num_data};
  NumArray<Real,1> num_data_copy(variant2);
  if (num_data_copy.to1DSpan()!=num_data.to1DSpan())
    ARCANE_FATAL("Bad value for copy");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
