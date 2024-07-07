// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtilsInternal.cc                                    (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires diverses sur les variables internes à Arcane.       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/VariableUtilsInternal.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/accelerator/core/RunQueue.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableUtilsInternal::
fillFloat64Array(IVariable* v, ArrayView<double> values)
{
  IData* var_data = v->data();
  auto* true_data = dynamic_cast<IArrayDataT<double>*>(var_data);
  if (!true_data)
    return true;
  // TODO: Vérifier la taille
  ArrayView<Real> var_values(true_data->view());
  values.copy(var_values);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableUtilsInternal::
setFromFloat64Array(IVariable* v, ConstArrayView<double> values)
{
  return setFromMemoryBuffer(v, ConstMemoryView(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableUtilsInternal::
setFromMemoryBuffer(IVariable* v, ConstMemoryView mem_view)
{
  INumericDataInternal* num_data = v->data()->_commonInternal()->numericData();
  if (!num_data)
    return true;
  RunQueue queue;
  impl::copyContiguousData(num_data, mem_view, queue);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDataInternal* VariableUtilsInternal::
getDataInternal(IVariable* v)
{
  return v->data()->_commonInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VariableUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
