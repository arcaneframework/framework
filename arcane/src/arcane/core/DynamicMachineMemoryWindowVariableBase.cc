// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMachineMemoryWindowVariableBase.h                 (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe DynamicMachineMemoryWindowBase.    */
/*---------------------------------------------------------------------------*/

#include "arcane/core/DynamicMachineMemoryWindowVariableBase.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IData.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/DynamicMachineMemoryWindowBase.h"
#include "arcane/core/IVariable.h"

#include "arcane/core/internal/DynamicMachineMemoryWindowMemoryAllocator.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arccore/common/AllocatedMemoryInfo.h"
#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMachineMemoryWindowVariableBase::
DynamicMachineMemoryWindowVariableBase(IVariable* var)
: m_var(var)
{
  if (!(m_var->property() & IVariable::PInShMem)) {
    ARCANE_FATAL("The variable has not PInShMem property");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> DynamicMachineMemoryWindowVariableBase::
segmentView() const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return DynamicMachineMemoryWindowMemoryAllocator::segmentView(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> DynamicMachineMemoryWindowVariableBase::
segmentView(Int32 rank) const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return DynamicMachineMemoryWindowMemoryAllocator::segmentView(data, rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> DynamicMachineMemoryWindowVariableBase::
machineRanks() const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return DynamicMachineMemoryWindowMemoryAllocator::machineRanks(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMachineMemoryWindowVariableBase::
barrier() const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  DynamicMachineMemoryWindowMemoryAllocator::barrier(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
