// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableBase.h                               (C) 2000-2026 */
/*                                                                           */
/* Allocateur mémoire utilisant la classe MachineShMemWinBase.               */
/*---------------------------------------------------------------------------*/

#include "arcane/core/MachineShMemWinVariableBase.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/ContigMachineShMemWin.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/IData.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariable.h"

#include "arcane/core/internal/MachineShMemWinMemoryAllocator.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arccore/common/AllocatedMemoryInfo.h"
#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableBase::
MachineShMemWinVariableBase(IVariable* var, Int64 sizeof_type)
: m_var(var)
, m_sizeof_type(sizeof_type)
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
  m_size_var.resize(m_pm->commSize(), 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> MachineShMemWinVariableBase::
machineRanks() const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return MachineShMemWinMemoryAllocator::machineRanks(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableBase::
barrier() const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  MachineShMemWinMemoryAllocator::barrier(data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinVariableBase::
segmentView() const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return MachineShMemWinMemoryAllocator::segmentView(data).subSpan(0, m_size_var[m_pm->commRank()]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Span<std::byte> MachineShMemWinVariableBase::
segmentView(Int32 rank) const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return MachineShMemWinMemoryAllocator::segmentView(data, rank).subSpan(0, m_size_var[rank]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableBase::
updateVariable(Int64 dim1)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 1);

  all_size.segmentView()[0] = dim1 * m_sizeof_type;
  all_size.barrier();
  m_size_var = all_size.windowConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* MachineShMemWinVariableBase::
variable() const
{
  return m_var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariable2DBase::
MachineShMemWinVariable2DBase(IVariable* var, Int64 sizeof_type)
: MachineShMemWinVariableBase(var, sizeof_type)
, m_dim1_var(m_pm->commSize(), 0)
, m_dim2_var(m_pm->commSize(), 0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariable2DBase::
updateVariable(Int64 dim1, Int64 dim2)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 2);

  all_size.segmentView()[0] = dim1 * m_sizeof_type;
  all_size.segmentView()[1] = dim2 * m_sizeof_type;

  all_size.barrier();

  for (Integer i = 0; i < m_pm->commSize(); ++i) {
    m_dim1_var[i] = all_size.windowConstView()[i * 2];
    m_dim2_var[i] = all_size.windowConstView()[i * 2 + 1];
    m_size_var[i] = m_dim1_var[i] * m_dim2_var[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <Int32 Dim>
MachineShMemWinVariableMDBase<Dim>::
MachineShMemWinVariableMDBase(IVariable* var, Int64 sizeof_type)
: MachineShMemWinVariableBase(var, sizeof_type)
, m_dim1_var(m_pm->commSize(), 0)
{}

template <Int32 Dim> void
MachineShMemWinVariableMDBase<Dim>::
updateVariable(Int64 dim1, SmallSpan<Int64, Dim> mdim)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 1);

  all_size.segmentView()[0] = dim1 * m_sizeof_type;

  all_size.barrier();

  m_dim1_var = all_size.windowConstView();
  m_mdim_var.span().copy(mdim);

  Int64 mult = 1;
  for (Integer i = 0; i < Dim; ++i) {
    mult *= mdim[i];
  }

  for (Integer i = 0; i < m_pm->commSize(); ++i) {
    m_size_var[i] = m_dim1_var[i] * mult;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MachineShMemWinVariableMDBase<1>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
