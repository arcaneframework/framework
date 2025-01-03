// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableDiff.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Gestion des différences entre les variables                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableDiff.h"

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IVariable.h"
#include "arcane/core/IParallelMng.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

template <typename DataType> void VariableDiff<DataType>::DiffPrinter::
sort(ArrayView<DiffInfo> diffs_info)
{
  if constexpr (std::is_same<TrueType, typename VarDataTypeTraits::IsNumeric>::value) {
    std::sort(std::begin(diffs_info), std::end(diffs_info));
  }
  else
    ARCANE_UNUSED(diffs_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> void VariableDiff<DataType>::DiffPrinter::
dump(ConstArrayView<DiffInfo> diffs_info, IVariable* var, IParallelMng* pm, int max_print)
{
  ITraceMng* msg = pm->traceMng();
  Int32 sid = pm->commRank();
  const String& var_name = var->name();
  Integer nb_diff = diffs_info.size();
  Integer nb_print = nb_diff;
  if (max_print >= 0 && nb_diff > static_cast<Integer>(max_print))
    nb_print = max_print;
  OStringStream ostr;
  ostr().precision(FloatInfo<Real>::maxDigit());
  ostr() << nb_diff << " entities having different values for the variable "
         << var_name << '\n';
  for (Integer i = 0; i < nb_print; ++i) {
    const DiffInfo& di = diffs_info[i];
    if (di.m_unique_id != NULL_ITEM_UNIQUE_ID) {
      // Il s'agit d'une entité
      char type = di.m_is_own ? 'O' : 'G';
      ostr() << "VDIFF: Variable '" << var_name << "'"
             << " (" << type << ")"
             << " uid=" << di.m_unique_id
             << " lid=" << di.m_local_id;
      if (di.m_sub_index != NULL_ITEM_ID)
        ostr() << " [" << di.m_sub_index << "]";
      ostr() << " val: " << di.m_current
             << " ref: " << di.m_ref << " rdiff: " << di.m_diff << '\n';
    }
    else {
      // Il s'agit de l'indice d'une variable tableau
      ostr() << "VDIFF: Variable '" << var_name << "'"
             << " index=" << di.m_local_id;
      if (di.m_sub_index != NULL_ITEM_ID)
        ostr() << " [" << di.m_sub_index << "]";
      ostr() << " val: " << di.m_current
             << " ref: " << di.m_ref << " rdiff: " << di.m_diff << '\n';
    }
  }
  msg->pinfo() << "Processor " << sid << " : " << nb_diff
               << " values are different on the variable "
               << var_name << ":\n"
               << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class VariableDiff<Byte>::DiffPrinter;
template class VariableDiff<Real>::DiffPrinter;
template class VariableDiff<Int8>::DiffPrinter;
template class VariableDiff<Int16>::DiffPrinter;
template class VariableDiff<Int32>::DiffPrinter;
template class VariableDiff<Int64>::DiffPrinter;
template class VariableDiff<BFloat16>::DiffPrinter;
template class VariableDiff<Float16>::DiffPrinter;
template class VariableDiff<Float32>::DiffPrinter;
template class VariableDiff<Real2>::DiffPrinter;
template class VariableDiff<Real2x2>::DiffPrinter;
template class VariableDiff<Real3>::DiffPrinter;
template class VariableDiff<Real3x3>::DiffPrinter;
template class VariableDiff<String>::DiffPrinter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
