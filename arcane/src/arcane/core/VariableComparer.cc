// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableComparer.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Classe pour effectuer des comparaisons entre les variables.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableComparer.h"

#include "arcane/core/IVariable.h"
#include "arcane/core/internal/IVariableInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableComparerArgs VariableComparer::
buildForCheckIfSync()
{
  VariableComparerArgs compare_args;
  compare_args.setCompareMode(eVariableComparerCompareMode::Sync);
  return compare_args;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableComparerArgs VariableComparer::
buildForCheckIfSameOnAllReplica()
{
  VariableComparerArgs compare_args;
  compare_args.setCompareMode(eVariableComparerCompareMode::SameOnAllReplica);
  return compare_args;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableComparerArgs VariableComparer::
buildForCheckIfSame(IDataReader* data_reader)
{
  ARCANE_CHECK_POINTER(data_reader);
  VariableComparerArgs compare_args;
  compare_args.setCompareMode(eVariableComparerCompareMode::Same);
  compare_args.setDataReader(data_reader);
  return compare_args;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableComparerResults VariableComparer::
apply(IVariable* var, const VariableComparerArgs& compare_args)
{
  ARCANE_CHECK_POINTER(var);
  return var->_internalApi()->compareVariable(compare_args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
