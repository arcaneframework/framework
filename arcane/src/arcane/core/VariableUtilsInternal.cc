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

#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"

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
  ArrayView<Real> var_values(true_data->view());
  values.copy(var_values);
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
