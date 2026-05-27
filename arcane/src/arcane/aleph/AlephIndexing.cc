// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephIndexing.cc                                            (C) 2000-2023 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "AlephArcane.h"

#include <map>
#include "arcane/IMesh.h"
#include "arcane/VariableInfo.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/utils/String.h"
#include <arcane/IVariable.h>
#include <arcane/IVariableMng.h>
#include <arcane/utils/ArcaneGlobal.h>
#include <arcane/utils/ArcanePrecomp.h>

#define ALEPH_INDEX_NOT_USED (-1)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// ****************************************************************************
// * AlephIndexing
// ****************************************************************************
AlephIndexing::
AlephIndexing(AlephKernel* kernel)
: TraceAccessor(kernel->parallel()->traceMng())
, m_kernel(kernel)
, m_sub_domain(kernel->subDomain())
, m_current_idx(0)
, m_known_items_own(0)
{
  debug() << "\33[1;33m[AlephIndexing::AlephIndexing] NEW"
          << "\33[m";
  m_known_items_all_address.resize(0);
}

// ****************************************************************************
// * updateKnownItems
// ****************************************************************************
Int32 AlephIndexing::
updateKnownItems(VariableItemInt32* var_idx,
                 const Item& itm)
{
  // In all cases, the address is this one
  m_known_items_all_address.add(&(*var_idx)[itm]);
  // If the targeted item is not ours, we should not count it
  if (itm.isOwn()) {
    // We update the memory slot, and we add it to the owned ones
    (*var_idx)[itm] = m_current_idx;
    m_known_items_own += 1;
  }
  else {
    //debug()<< "\t\t\33[33m[AlephIndexing::updateKnownItems] " << var_idx->name() << " is NOT ours"<<"\33[m";
    (*var_idx)[itm] = m_current_idx;
  }
  // Now, we can increment the row index
  m_current_idx += 1;
  // And we return the requested row number
  //debug()<< "\t\t\33[33m[AlephIndexing::updateKnownItems] returning \33[1;32m"<<m_current_idx-1<<"\33[m";
  return m_current_idx - 1;
}

// ****************************************************************************
// * findWhichLidFromMapMap
// ****************************************************************************
Int32 AlephIndexing::
findWhichLidFromMapMap(IVariable* var,
                       const Item& itm)
{
#ifdef ARCANE_CHECK
  if (itm.null())
    ARCANE_FATAL("Null item");
#endif

  VarMapIdx::const_iterator iVarMap = m_var_map_idx.find(var);
  // If the variable is not even known yet
  // We add a map entry (map(m_current_idx))
  if (iVarMap == m_var_map_idx.end()) {
    //debug()<<"\t\33[33m[findWhichLidFromMapMap] Unknown variable "<<var->name()<<"\33[m";
    traceMng()->flush();
    String var_idx_name(var->name());
    var_idx_name = var_idx_name + String("_idx");
    VariableItemInt32* var_idx =
    new VariableItemInt32(VariableBuildInfo(var->itemFamily(),
                                            var_idx_name,IVariable::PSubDomainDepend),
                          var->itemKind());
    // We add the '_idx' variable of this variable to our map
    m_var_map_idx.insert(std::make_pair(var, var_idx));
    // We flush all potential indices of this variable
    var_idx->fill(ALEPH_INDEX_NOT_USED);
    return updateKnownItems(var_idx, itm);
  }
  VariableItemInt32* var_idx = iVarMap->second;
  // If this item is not known by this variable, we add an entry
  if ((*var_idx)[itm] == ALEPH_INDEX_NOT_USED) {
    //debug()<<"\t\33[33m[findWhichLidFromMapMap] This item is not known by this variable, we add an entry\33[m";
    traceMng()->flush();
    return updateKnownItems(var_idx, itm);
  }
  //debug()<<"\t\33[33m[AlephIndexing::findWhichLidFromMapMap] " <<var->name()<<" "<<var->itemKind() << " hits row #\33[1;32m"<<(*var_idx)[itm]<<"\33[m";
  traceMng()->flush();
  return (*var_idx)[itm];
}

// ****************************************************************************
// * get which triggers findWhichLidFromMapMap
// ****************************************************************************
Int32 AlephIndexing::
get(const VariableRef& variable,
    const ItemEnumerator& itm)
{
  return get(variable, *itm);
}
Int32 AlephIndexing::get(const VariableRef& variable,
                         const Item& itm)
{
  IVariable* var = variable.variable();
  if (m_kernel->isInitialized()){
    auto x = m_var_map_idx.find(var);
    if (x==m_var_map_idx.end())
      ARCANE_FATAL("Can not find variable {0}",var->name());
    return (*x->second)[itm] - m_kernel->topology()->part()[m_kernel->rank()];
  }
  // We test if we are working on a scalar variable
  if (var->dimension() != 1)
    throw ArgumentException(A_FUNCINFO, "cannot get non-scalar variables!");
  // We check that the item type is well known
  if (var->itemKind() >= IK_Unknown)
    throw ArgumentException(A_FUNCINFO, "Unknown Item Kind!");
  //debug()<<"\33[1;33m[AlephIndexing::get] Valid couple, now looking for known idx (uid="<<itm->uniqueId()<<")\33[m";
  return findWhichLidFromMapMap(var, itm);
}

// ****************************************************************************
// * buildIndexesFromAddress
// ****************************************************************************
void AlephIndexing::
buildIndexesFromAddress(void)
{
  const Integer topology_row_offset = m_kernel->topology()->part()[m_kernel->rank()];
  VarMapIdx::const_iterator iVarIdx = m_var_map_idx.begin();
  debug() << "\33[1;7;33m[buildIndexesFromAddress] Re-indexing variables with offset " << topology_row_offset << "\33[m";
  // We re-index and synchronize all variables that we have seen
  for (; iVarIdx != m_var_map_idx.end(); ++iVarIdx) {
    ItemGroup group = iVarIdx->first->itemGroup();
    VariableItemInt32* var_idx = iVarIdx->second;
    ENUMERATE_ITEM (itm, group) {
      // If this item is not used, we skip it
      if ((*var_idx)[itm] == ALEPH_INDEX_NOT_USED)
        continue;
      // Otherwise, we add the offset
      (*var_idx)[itm] += topology_row_offset;
    }
    debug() << "\t\33[1;7;33m[buildIndexesFromAddress] Synchronizing idx for variable " << iVarIdx->second->name() << "\33[m";
    iVarIdx->second->synchronize();
  }
}

// ****************************************************************************
// * localKnownItems
// * Consolidation of m_known_items_own based on items
// ****************************************************************************
Integer AlephIndexing::
localKnownItems(void)
{
  return m_known_items_own;
}

// ****************************************************************************
// * nowYouCanBuildTheTopology
// ****************************************************************************
void AlephIndexing::
nowYouCanBuildTheTopology(AlephMatrix* fromThisMatrix,
                          AlephVector* fromThisX,
                          AlephVector* fromThisB)
{
  // Retrieval of the item count consolidation
  Integer lki = localKnownItems();
  // ReduceSum over the entire topology
  Integer gki = m_kernel->parallel()->reduce(Parallel::ReduceSum, lki);
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] Working with lki="
          << lki << ", gki=" << gki << "\33[m";
  // Initialization of the Aleph kernel based on local and global known items
  m_kernel->initialize(gki, lki);
  // From here, the kernel is initialized, we use m_arguments_queue directly
  // The topology has been replaced by a new one
  debug() << "\33[1;7;33m[AlephIndexing::nowYouCanBuildTheTopology] Kernel is now initialized, rewinding Aleph operations!\33[m";
  // If we are in parallel, we must consolidate the indices according to the new topology
  if (m_kernel->isParallel())
    buildIndexesFromAddress();
  // We can now create the triplet (matrix,lhs,rhs)
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] asking kernel for a Matrix\33[m";
  AlephMatrix* firstMatrix = m_kernel->createSolverMatrix();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] asking kernel for a RHS Vector\33[m";
  AlephVector* firstRhsVector = m_kernel->createSolverVector();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] asking kernel for a LHS Vector\33[m";
  AlephVector* firstLhsVector = m_kernel->createSolverVector();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstMatrix->create()\33[m";
  firstMatrix->create();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstRhsVector->create()\33[m";
  firstRhsVector->create();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstLhsVector->create()\33[m";
  firstLhsVector->create();
  // And we return to replay the matrix's setValues with the consolidated indices
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] reSetValues fromThisMatrix\33[m";
  fromThisMatrix->reSetValuesIn(firstMatrix,
                                m_known_items_all_address);
  // And we return to replay the matrix's addValues with the consolidated indices
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] reAddValues fromThisMatrix\33[m";
  fromThisMatrix->reAddValuesIn(firstMatrix,
                                m_known_items_all_address);
  // We re-run the assembly
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstMatrix->assemble()\33[m";
  firstMatrix->assemble();
  // And we do the same process for lhs and rhs
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstRhsVector reSetLocalComponents/assemble\33[m";
  firstRhsVector->reSetLocalComponents(fromThisB);
  firstRhsVector->assemble();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstLhsVector reSetLocalComponents/assemble\33[m";
  firstLhsVector->reSetLocalComponents(fromThisX);
  firstLhsVector->assemble();
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] nothing more to do here!\33[m";
}

// ****************************************************************************
// * ~AlephIndexing
// ****************************************************************************
AlephIndexing::
~AlephIndexing()
{
  debug() << "\t\33[1;33m[AlephIndexing::~AlephIndexing] deleting each new'ed VarMapIdx..."
          << "\33[m";
  VarMapIdx::const_iterator iVarIdx = m_var_map_idx.begin();
  for (; iVarIdx != m_var_map_idx.end(); ++iVarIdx)
    delete iVarIdx->second;
  debug() << "\t\33[1;33m[AlephIndexing::~AlephIndexing] done!"
          << "\33[m";
  traceMng()->flush();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
