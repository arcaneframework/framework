// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MachineShMemWinVariableBase.cc                              (C) 2000-2026 */
/*                                                                           */
/* Classes de bases permettant d'exploiter l'objet MachineShMemWinVariable   */
/* pointé de la zone mémoire des variables en mémoire partagée.              */
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
  m_sizeof_var.resize(m_pm->commSize(), 0);
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
segmentView(Int32 rank) const
{
  const AllocatedMemoryInfo data(m_var->data()->_commonInternal()->numericData()->memoryView().data());
  return MachineShMemWinMemoryAllocator::segmentView(data, rank).subSpan(0, m_sizeof_var[rank]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableBase::
updateVariable(Int64 nb_elem_dim1, Int64 sizeof_elem)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 1);

  all_size.segmentView()[0] = nb_elem_dim1 * sizeof_elem;
  all_size.barrier();
  m_sizeof_var = all_size.windowConstView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariable2DBase::
MachineShMemWinVariable2DBase(IVariable* var)
: MachineShMemWinVariableBase(var)
, m_nb_elem_dim1(m_pm->commSize(), 0)
, m_nb_elem_dim2(m_pm->commSize(), 0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariable2DBase::
updateVariable(Int64 nb_elem_dim1, Int64 nb_elem_dim2, Int64 sizeof_elem)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 2);

  all_size.segmentView()[0] = nb_elem_dim1;
  all_size.segmentView()[1] = nb_elem_dim2;

  all_size.barrier();

  Int64 sizeof_elem2 = sizeof_elem * sizeof_elem;

  for (Int32 i = 0; i < m_pm->commSize(); ++i) {
    const Int32 i2 = i * 2;
    m_nb_elem_dim1[i] = all_size.windowConstView()[i2];
    m_nb_elem_dim2[i] = all_size.windowConstView()[i2 + 1];
    m_sizeof_var[i] = m_nb_elem_dim1[i] * m_nb_elem_dim2[i] * sizeof_elem2;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayView<Int64> MachineShMemWinVariable2DBase::
nbElemDim1()
{
  return m_nb_elem_dim1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayView<Int64> MachineShMemWinVariable2DBase::
nbElemDim2()
{
  return m_nb_elem_dim2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MachineShMemWinVariableMDBase::
MachineShMemWinVariableMDBase(IVariable* var)
: MachineShMemWinVariableBase(var)
, m_nb_elem_dim1(m_pm->commSize(), 0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MachineShMemWinVariableMDBase::
updateVariable(Int64 nb_elem_dim1, Int32 nb_elem_dim2, Int64 sizeof_elem)
{
  ContigMachineShMemWin<Int64> all_size(m_pm, 1);

  all_size.segmentView()[0] = nb_elem_dim1;

  all_size.barrier();

  m_nb_elem_dim1 = all_size.windowConstView();

  Int64 mult = nb_elem_dim2 * sizeof_elem * sizeof_elem;

  for (Integer i = 0; i < m_pm->commSize(); ++i) {
    m_sizeof_var[i] = m_nb_elem_dim1[i] * mult;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayView<Int64> MachineShMemWinVariableMDBase::
nbElemDim1()
{
  return m_nb_elem_dim1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayShape MachineShMemWinVariableMDBase::
arrayShape()
{
  return m_var->data()->shape();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
