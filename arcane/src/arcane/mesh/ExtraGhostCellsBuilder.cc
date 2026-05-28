// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostCellsBuilder.cc                                   (C) 2000-2024 */
/*                                                                           */
/* Construction of extra ghost cells.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IExtraGhostCellsBuilder.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/mesh/ExtraGhostCellsBuilder.h"
#include "arcane/mesh/DynamicMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExtraGhostCellsBuilder::
ExtraGhostCellsBuilder(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostCellsBuilder::
addExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder)
{
  if (m_builders.contains(builder))
    ARCANE_FATAL("Instance {0} is already registered", builder);
  m_builders.add(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostCellsBuilder::
removeExtraGhostCellsBuilder(IExtraGhostCellsBuilder* builder)
{
  auto iter_begin = m_builders.begin();
  auto iter_end = m_builders.end();
  auto x = std::find(iter_begin, iter_end, builder);
  if (x == iter_end)
    ARCANE_FATAL("Instance {0} is not registered", builder);
  m_builders.remove(x - iter_begin);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ExtraGhostCellsBuilder::
hasBuilder() const
{
  return !m_builders.empty();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostCellsBuilder::
computeExtraGhostCells()
{
  if (m_builders.empty())
    return;

  info() << "Compute extra ghost cells";

  for (IExtraGhostCellsBuilder* v : m_builders) {
    // Calculate extra cells to send
    v->computeExtraCellsToSend();
  }

  IParallelMng* pm = m_mesh->parallelMng();

  auto exchanger{ ParallelMngUtils::createExchangerRef(pm) };

  const Int32 nsd = pm->commSize();

  // Construction of items to send
  UniqueArray<std::set<Integer>> to_sends(nsd);

  // Initialisation of the data exchanger
  for (Integer isd = 0; isd < nsd; ++isd) {
    std::set<Integer>& cell_set = to_sends[isd];
    for (IExtraGhostCellsBuilder* builder : m_builders) {
      Int32ConstArrayView extra_cells = builder->extraCellsToSend(isd);
      // We sort the IDs to send to avoid duplicates
      for (Integer j = 0, size = extra_cells.size(); j < size; ++j)
        cell_set.insert(extra_cells[j]);
    }
    if (!cell_set.empty())
      exchanger->addSender(isd);
  }
  exchanger->initializeCommunicationsMessages();

  // Sending the cells
  for (Integer i = 0, ns = exchanger->nbSender(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToSend(i);
    const Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    const std::set<Integer>& cell_set = to_sends[rank];
    Int32UniqueArray items_to_send(cell_set.size());
    std::copy(std::begin(cell_set), std::end(cell_set), std::begin(items_to_send));
    m_mesh->serializeCells(s, items_to_send);
  }
  exchanger->processExchange();

  // Receiving the cells
  for (Integer i = 0, ns = exchanger->nbReceiver(); i < ns; ++i) {
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* s = sm->serializer();
    m_mesh->addCells(s);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
