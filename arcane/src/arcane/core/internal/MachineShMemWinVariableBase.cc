// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableBase.cc                              (C) 2000-2026 */
/*                                                                           */
/* Base classes allowing the exploitation of the MachineShMemWinVariable     */
/* object pointed to the memory area of variables in shared memory.          */
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/MachineShMemWinVariableBase.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayShape.h"

#include "arcane/core/ContigMachineShMemWin.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IData.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariable.h"

#include "arcane/core/internal/MachineShMemWinMemoryAllocator.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IParallelMngInternal.h"

#include "arccore/common/AllocatedMemoryInfo.h"
#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableBase::
MachineShMemWinVariableBase(IVariable* var)
: m_var(var)
{
  if (!(m_var->property() & IVariable::PInShMem)) {
    ARCANE_FATAL("The variable has not PInShMem property");
  }
  if (m_var->meshHandle().hasMesh()) {
    m_pm = m_var->meshHandle().mesh()->parallelMng();
  }
  else {
    m_pm = m_var->subDomain()->parallelMng();
  }
  m_machine_ranks = m_pm->_internalApi()->machineRanks();
  m_sizeof_var.reserve(m_machine_ranks.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MachineShMemWinVariableBase::
machineRanks() const
{
  return m_machine_ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableBase::
barrier() const
{
  return m_pm->_internalApi()->machineBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinVariableBase::
segmentView(Int32 rank) const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
#ifdef ARCANE_CHECK
  if (data.baseAddress() == nullptr) {
    ARCANE_FATAL("Variable not initialised yet. Call var.resize() method before.");
  }
#endif
  return MachineShMemWinMemoryAllocator::segmentView(data, rank).subSpan(0, m_sizeof_var.at(rank));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableBase::
updateVariable(Int64 nb_elem_dim1, Int64 sizeof_elem)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 1);

  all_size.segmentView()[0] = nb_elem_dim1 * sizeof_elem;
  all_size.barrier();

  for (Int32 machine_rank = 0; const Int64 size : all_size.windowConstView()) {
    m_sizeof_var[m_machine_ranks[machine_rank]] = size;
    machine_rank++;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariable2DBase::
MachineShMemWinVariable2DBase(IVariable* var)
: MachineShMemWinVariableBase(var)
{
  m_nb_elem_dim1.reserve(m_machine_ranks.size());
  m_nb_elem_dim2.reserve(m_machine_ranks.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariable2DBase::
updateVariable(Int64 nb_elem_dim1, Int64 nb_elem_dim2, Int64 sizeof_elem)
{
  ContigMachineShMemWin<Int64> all_nb_elem(m_pm, 2);

  all_nb_elem.segmentView()[0] = nb_elem_dim1;
  all_nb_elem.segmentView()[1] = nb_elem_dim2;

  all_nb_elem.barrier();

  Int64 sizeof_elem2 = sizeof_elem * sizeof_elem;

  for (Int32 machine_rank = 0; const Int32 world_rank : m_machine_ranks) {
    const Int32 pos = machine_rank * 2;
    const Int64 nb_elem_dim1_i = all_nb_elem.windowConstView()[pos];
    const Int64 nb_elem_dim2_i = all_nb_elem.windowConstView()[pos + 1];

    m_nb_elem_dim1[world_rank] = nb_elem_dim1_i;
    m_nb_elem_dim2[world_rank] = nb_elem_dim2_i;
    m_sizeof_var[world_rank] = nb_elem_dim1_i * nb_elem_dim2_i * sizeof_elem2;
    machine_rank++;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableMDBase::
MachineShMemWinVariableMDBase(IVariable* var)
: MachineShMemWinVariableBase(var)
{
  m_nb_elem_dim1.reserve(m_machine_ranks.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableMDBase::
updateVariable(Int64 nb_elem_dim1, Int32 nb_elem_dim2, Int64 sizeof_elem)
{
  ContigMachineShMemWin<Int64> all_nb_elem(m_pm, 1);

  all_nb_elem.segmentView()[0] = nb_elem_dim1;

  all_nb_elem.barrier();

  Int64 mult = nb_elem_dim2 * sizeof_elem * sizeof_elem;

  for (Int32 machine_rank = 0; const auto nb_elem : all_nb_elem.windowConstView()) {
    const Int32 world_rank = m_machine_ranks[machine_rank];
    m_nb_elem_dim1[world_rank] = nb_elem;
    m_sizeof_var[world_rank] = nb_elem * mult;
    machine_rank++;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayShape MachineShMemWinVariableMDBase::
arrayShape() const
{
  return m_var->data()->shape();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
