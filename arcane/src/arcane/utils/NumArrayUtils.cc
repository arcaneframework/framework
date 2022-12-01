// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayUtils.cc                                            (C) 2000-2022 */
/*                                                                           */
/* Fonctions utilitaires pour NumArray.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArrayUtils.h"
#include "arcane/utils/NumArray.h"
#include "arcane/utils/IOException.h"

#include "arcane/utils/internal/ValueConvertInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \namespace Arcane::NumArrayUtils
 *
 * \brief Fonctions utilitaires pour NumArray.
 */
namespace Arcane::NumArrayUtils
{

namespace
{
template<typename DataType> void
_readFromText(NumArray<DataType, MDDim1>& num_array, std::istream& input)
{
  UniqueArray<DataType> v;
  if (builtInGetArrayValueFromStream(v,input))
    ARCANE_THROW(IOException,"Error filling NumArray with text file");
  MDSpan<DataType,MDDim1> data(v.data(),v.size());
  num_array.resize(v.size());
  num_array.copy(data);
}

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Remplit \a v avec les valeurs de input.
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<double, MDDim1>& v, std::istream& input)
{
  _readFromText(v,input);
}

//! Remplit \a v avec les valeurs de input.
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<Int32, MDDim1>& v, std::istream& input)
{
  _readFromText(v,input);
}

//! Remplit \a v avec les valeurs de input.
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<Int64, MDDim1>& v, std::istream& input)
{
  _readFromText(v,input);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::NumArrayUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
