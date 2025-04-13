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

VariableComparer::
VariableComparer(ITraceMng* tm)
: TraceAccessor(tm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 VariableComparer::
checkIfSync(IVariable* var, Int32 max_print)
{
  ARCANE_CHECK_POINTER(var);
  return var->checkIfSync(max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 VariableComparer::
checkIfSame(IVariable* var, IDataReader* reader, Int32 max_print, bool compare_ghost)
{
  ARCANE_CHECK_POINTER(var);
  return var->checkIfSame(reader, max_print, compare_ghost);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 VariableComparer::
checkIfSameOnAllReplica(IVariable* var, Integer max_print)
{
  ARCANE_CHECK_POINTER(var);
  return var->checkIfSameOnAllReplica(max_print);
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
