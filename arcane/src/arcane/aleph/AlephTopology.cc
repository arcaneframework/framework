// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephTopology.cc                                            (C) 2000-2024 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephTopology.h"

#include "arcane/aleph/AlephKernel.h"
#include "arcane/aleph/AlephArcane.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// *****************************************************************************
// * Minimal AlephTopology for AlephIndexing
// *****************************************************************************
AlephTopology::
AlephTopology(AlephKernel* kernel)
: TraceAccessor(kernel->parallel()->traceMng())
, m_kernel(kernel)
, m_nb_row_size(0)
, m_nb_row_rank(0)
, m_gathered_nb_row_elements(0)
, m_created(false)
, m_has_set_row_nb_elements(false)
, m_has_been_initialized(false)
{
  debug() << "\33[1;32m\t[AlephTopology::AlephTopology] Loading MINIMALE AlephTopology"
          << "\33[0m";
}

/******************************************************************************
 *****************************************************************************/
AlephTopology::
AlephTopology(ITraceMng* tm,
              AlephKernel* kernel,
              Integer nb_row_size,
              Integer nb_row_rank)
: TraceAccessor(tm)
, m_kernel(kernel)
, m_nb_row_size(nb_row_size)
, m_nb_row_rank(nb_row_rank)
, m_gathered_nb_row_elements()
, m_created(false)
, m_has_set_row_nb_elements(false)
, m_has_been_initialized(true)
{
  ItacFunction(AlephTopology);
  debug() << "\33[1;32m\t[AlephTopology::AlephTopology] Loading AlephTopology"
          << "\33[0m";

  m_gathered_nb_setValued.resize(m_kernel->size());
  m_gathered_nb_row.resize(m_kernel->size() + 1);
  m_gathered_nb_row[0] = 0;

  if (!m_kernel->isParallel()) {
    m_gathered_nb_row[1] = m_nb_row_size;
    debug() << "\33[1;32m\t[AlephTopology::AlephTopology] SEQ done"
            << "\33[0m";
    return;
  }

  if (m_kernel->isAnOther()) {
    debug() << "\33[1;32m\t[AlephTopology::AlephTopology] receiving m_gathered_nb_row"
            << "\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_row.view(), 0);
    return;
  }

  debug() << "\33[1;32m\t[AlephTopology::AlephTopology] Nous nous échangeons les indices locaux des lignes de la matrice"
          << "\33[0m";
  UniqueArray<Integer> all_rows;
  UniqueArray<Integer> gathered_nb_row(m_kernel->size());
  all_rows.add(m_nb_row_rank);
  m_kernel->parallel()->allGather(all_rows, gathered_nb_row);
  for (int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
    m_gathered_nb_row[iCpu + 1] = m_gathered_nb_row[iCpu] + gathered_nb_row[iCpu];
    debug() << "\33[1;32m\t\t[AlephTopology::AlephTopology] " << iCpu << ":" << m_gathered_nb_row[iCpu] << "\33[0m";
  }
  debug() << "\33[1;32m\t[AlephTopology::AlephTopology] m_parallel_info_partitioning done"
          << "\33[0m";

  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
    debug() << "\33[1;32m\t[AlephTopology::AlephTopology] sending m_gathered_nb_row"
            << "\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_row.view(), 0);
  }
}

/******************************************************************************
 *****************************************************************************/
AlephTopology::
~AlephTopology()
{
  debug() << "\33[1;5;32m\t[~AlephTopology]"
          << "\33[0m";
}

/******************************************************************************
 * b1e13efe
 *****************************************************************************/
void AlephTopology::
create(Integer setValue_idx)
{
  ItacFunction(AlephTopology);

  if (m_created)
    return;
  m_created = true;

  checkForInit();

  if (!m_kernel->isParallel()) {
    debug() << "\33[1;32m\t\t\t[AlephTopology::create] SEQ m_gathered_nb_setValued[0]=" << setValue_idx << "\33[0m";
    m_gathered_nb_setValued[0] = setValue_idx;
    return;
  }

  debug() << "\33[1;32m\t\t\t[AlephTopology::create]"
          << "\33[0m";
  if (m_kernel->isAnOther()) {
    debug() << "\33[1;32m\t[AlephTopology::create] receiving m_gathered_nb_setValued"
            << "\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_setValued.view(), 0);
    return;
  }

  // Nous allons nous échanger tous les setValue_idx
  UniqueArray<AlephInt> all;
  all.add(setValue_idx);
  m_kernel->parallel()->allGather(all, m_gathered_nb_setValued);

  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
    debug() << "\33[1;32m\t[AlephTopology::create] sending m_gathered_nb_setValued"
            << "\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_setValued.view(), 0);
  }
  debug() << "\33[1;32m\t\t\t[AlephTopology::create] done"
          << "\33[0m";
}

/******************************************************************************
 * 1b264c6c
 * Ce row_nb_element est positionné afin d'aider à la construction de la matrice
 * lors des :
 *     - 'init_length' de Sloop
 *     - HYPRE_IJMatrixSetRowSizes
 *     - Trilinos Epetra_CrsMatrix
 *****************************************************************************/
void AlephTopology::
setRowNbElements(IntegerConstArrayView row_nb_element)
{
  checkForInit();

  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements]"
          << "\33[0m";
  if (m_has_set_row_nb_elements)
    return;
  m_has_set_row_nb_elements = true;
  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements]"
          << "\33[0m";

  // Nous allons nous échanger les nombre d'éléments par lignes
  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] resize m_gathered_nb_row_elements to " << m_nb_row_size << "\33[0m";
  m_gathered_nb_row_elements.resize(m_nb_row_size);

  if (m_kernel->isAnOther()) {
    debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] isAnOther from 0"
            << "\33[0m";
    traceMng()->flush();
    m_kernel->world()->broadcast(m_gathered_nb_row_elements.view(), 0);
    debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] done"
            << "\33[0m";
    traceMng()->flush();
    return;
  }

  UniqueArray<AlephInt> local_row_nb_element(m_nb_row_rank);
  for (int i = 0; i < m_nb_row_rank; ++i)
    local_row_nb_element[i] = row_nb_element[i];
  m_kernel->parallel()->allGatherVariable(local_row_nb_element, m_gathered_nb_row_elements);

  if (m_kernel->thereIsOthers() && !m_kernel->isAnOther()) {
    debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] Sending m_gathered_nb_row_elements of size=" << m_gathered_nb_row_elements.size() << "\33[0m";
    m_kernel->world()->broadcast(m_gathered_nb_row_elements.view(), 0);
  }
  debug() << "\33[1;32m\t\t\t[AlephTopology::setRowNbElements] done"
          << "\33[0m";
}

/******************************************************************************
 *****************************************************************************/
IntegerConstArrayView AlephTopology::
ptr_low_up_array()
{
  debug() << "\33[1;32m\t[AlephTopology::ptr_low_up_array]"
          << "\33[0m";
  return IntegerConstArrayView();
}

/******************************************************************************
 *****************************************************************************/
ConstArrayView<AlephInt> AlephTopology::
part()
{
  checkForInit();
  //debug() << "\33[1;32m\t[AlephTopology::part]"<<"\33[0m";
  return m_gathered_nb_row;
}

/******************************************************************************
 *****************************************************************************/
IParallelMng* AlephTopology::
parallelMng()
{
  debug() << "\33[1;32m\t[AlephTopology::parallelMng]"
          << "\33[0m";
  return m_kernel->parallel();
}

/******************************************************************************
 *****************************************************************************/
void AlephTopology::
rowRange(Integer& min_row, Integer& max_row)
{
  const Integer rank = m_kernel->rank();
  checkForInit();
  debug() << "\33[1;32m\t[AlephTopology::rowRange] rank=" << rank << "\33[0m";
  min_row = m_gathered_nb_row[rank];
  max_row = m_gathered_nb_row[rank + 1];
  debug() << "\33[1;32m\t[AlephTopology::rowRange] min_row=" << min_row << ", max_row=" << max_row << "\33[0m";
}

/******************************************************************************
 *****************************************************************************/
Integer AlephTopology::
rowLocalRange(const Integer index)
{
  Integer ilower = -1;
  Integer iupper = 0;
  Integer range = 0;
  checkForInit();
  for (int iCpu = 0; iCpu < m_kernel->size(); ++iCpu) {
    if (m_kernel->rank() != m_kernel->solverRanks(index)[iCpu])
      continue;
    if (ilower == -1)
      ilower = m_kernel->topology()->gathered_nb_row(iCpu);
    iupper = m_kernel->topology()->gathered_nb_row(iCpu + 1) - 1;
  }
  range = iupper - ilower + 1;
  debug() << "\33[1;32m\t[AlephTopology::rowLocalRange] ilower=" << ilower
          << ", iupper=" << iupper << ", range=" << range << "\33[0m";
  return range;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
