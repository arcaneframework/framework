// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SynchronizerMatrixPrinter.h                                 (C) 2011-2011 */
/*                                                                           */
/* Affiche la matrix de synchronization.                                     */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/IParallelMng.h"
#include "arcane/MathUtils.h"
#include "arcane/SynchronizerMatrixPrinter.h"

#include <iomanip>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SynchronizerMatrixPrinter::
SynchronizerMatrixPrinter(IVariableSynchronizer* synchronizer)
  : m_synchronizer(synchronizer)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
SynchronizerMatrixPrinter::
print(std::ostream & o) const
{
  IParallelMng * parallel_mng = m_synchronizer->parallelMng();  
  Int32ConstArrayView communicating_ranks = m_synchronizer->communicatingRanks();
  Int32UniqueArray shared_counts(communicating_ranks.size());
  for(Integer i=0;i<communicating_ranks.size();++i)
    shared_counts[i] = m_synchronizer->sharedItems(i).size();
  Int32UniqueArray ghost_counts(communicating_ranks.size());
  for(Integer i=0;i<communicating_ranks.size();++i)
    ghost_counts[i] = m_synchronizer->ghostItems(i).size();
  Int32UniqueArray comm_size(1);
  comm_size[0] = communicating_ranks.size();

  Int32UniqueArray all_comm_sizes(parallel_mng->commSize());
  parallel_mng->gather(comm_size,all_comm_sizes,0);
  Int32UniqueArray all_communicating_ranks;
  parallel_mng->gatherVariable(communicating_ranks,all_communicating_ranks,0);
  Int32UniqueArray all_shared_counts;
  parallel_mng->gatherVariable(shared_counts,all_shared_counts,0);
  Int32UniqueArray all_ghost_counts;
  parallel_mng->gatherVariable(ghost_counts,all_ghost_counts,0);
  ARCANE_ASSERT((all_ghost_counts.size() == all_shared_counts.size()),("Incompatible shared / ghost counts"));

  if (parallel_mng->commRank() == 0){
    Int32UniqueArray all_offsets(all_comm_sizes.size()+1);
    all_offsets[0] = 0;
    for(Integer i=0;i<all_comm_sizes.size();++i)
      all_offsets[i+1] = all_offsets[i] + all_comm_sizes[i];
      
    Int64 text_width0 = String::format("{0}",parallel_mng->commSize()).length();
    for(Integer i=0;i<all_shared_counts.size();++i)
      text_width0 = math::max(text_width0,String::format("{0} / {1}",all_shared_counts[i], all_ghost_counts[i]).length());
    text_width0 = math::max(6,text_width0);
    Int32 text_width  = CheckedConvert::toInt32(text_width0);
      
    o << "\n" << std::setw(text_width) << "" << "   ";
    for(Integer j=0;j<parallel_mng->commSize();++j)
      o << std::setw(text_width) << j << " ";
    o << "\n";

    for(Integer i=0;i<parallel_mng->commSize();++i) {
      o << std::setw(text_width) << i << " : ";
      Integer jstart = all_offsets[i];
      Integer jend = all_offsets[i+1];
      for(Integer j=0;j<parallel_mng->commSize();++j) {
        if (jstart != jend && all_communicating_ranks[jstart] == j) {
          o << std::setw(text_width) << String::format("{0} / {1}",all_shared_counts[jstart], all_ghost_counts[jstart]) << " ";
          ++jstart;
        }
        else
          o << std::setw(text_width) << "" << " ";
      }
      o << "\n";
    }
    o << "(i,j) = S/G : proc 'i' shares S items with 'j' / has G ghosts items from 'j'\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
