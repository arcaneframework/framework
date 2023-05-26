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
  // Dans tous les cas l'adresse est celle-ci
  m_known_items_all_address.add(&(*var_idx)[itm]);
  // Si l'item ciblé n'est pas à nous, on ne doit pas le compter
  if (itm.isOwn()) {
    // On met à jour la case mémoire, et on le rajoute aux owned
    (*var_idx)[itm] = m_current_idx;
    m_known_items_own += 1;
  }
  else {
    //debug()<< "\t\t\33[33m[AlephIndexing::updateKnownItems] " << var_idx->name() << " is NOT ours"<<"\33[m";
    (*var_idx)[itm] = m_current_idx;
  }
  // Maintenant, on peut incrémenter l'index de ligne
  m_current_idx += 1;
  // Et on retourne le numéro demandé de la ligne
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
  // Si la variable n'est même pas encore connue
  // On rajoute une entrée map(map(m_current_idx))
  if (iVarMap == m_var_map_idx.end()) {
    //debug()<<"\t\33[33m[findWhichLidFromMapMap] Unknown variable "<<var->name()<<"\33[m";
    traceMng()->flush();
    String var_idx_name(var->name());
    var_idx_name = var_idx_name + String("_idx");
    VariableItemInt32* var_idx =
    new VariableItemInt32(VariableBuildInfo(var->itemFamily(),
                                            var_idx_name,IVariable::PSubDomainDepend),
                          var->itemKind());
    // On rajoute à notre map la variable '_idx' de cette variable
    m_var_map_idx.insert(std::make_pair(var, var_idx));
    // On flush tous les indices potentiels de cette variable
    var_idx->fill(ALEPH_INDEX_NOT_USED);
    return updateKnownItems(var_idx, itm);
  }
  VariableItemInt32* var_idx = iVarMap->second;
  // Si cet item n'est pas connu de cette variable, on rajoute une entrée
  if ((*var_idx)[itm] == ALEPH_INDEX_NOT_USED) {
    //debug()<<"\t\33[33m[findWhichLidFromMapMap] Cet item n'est pas connu de cette variable, on rajoute une entrée\33[m";
    traceMng()->flush();
    return updateKnownItems(var_idx, itm);
  }
  //debug()<<"\t\33[33m[AlephIndexing::findWhichLidFromMapMap] " <<var->name()<<" "<<var->itemKind() << " hits row #\33[1;32m"<<(*var_idx)[itm]<<"\33[m";
  traceMng()->flush();
  return (*var_idx)[itm];
}

// ****************************************************************************
// * get qui trig findWhichLidFromMapMap
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
  // On teste de bien travailler sur une variables scalaire
  if (var->dimension() != 1)
    throw ArgumentException(A_FUNCINFO, "cannot get non-scalar variables!");
  // On vérifie que le type d'item est bien connu
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
  debug() << "\33[1;7;33m[buildIndexesFromAddress] Re-inexing variables with offset " << topology_row_offset << "\33[m";
  // On ré-indice et synchronise toutes les variables qu'on a pu voir passer
  for (; iVarIdx != m_var_map_idx.end(); ++iVarIdx) {
    ItemGroup group = iVarIdx->first->itemGroup();
    VariableItemInt32* var_idx = iVarIdx->second;
    ENUMERATE_ITEM (itm, group) {
      // Si cet item n'est pas utilisé, on s'en occupe pas
      if ((*var_idx)[itm] == ALEPH_INDEX_NOT_USED)
        continue;
      // Sinon on rajoute l'offset
      (*var_idx)[itm] += topology_row_offset;
    }
    debug() << "\t\33[1;7;33m[buildIndexesFromAddress] Synchronizing idx for variable " << iVarIdx->second->name() << "\33[m";
    iVarIdx->second->synchronize();
  }
}

// ****************************************************************************
// * localKnownItems
// * Consolidation en nombre des m_known_items_own fonction des items
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
  // Récupération de la consolidation en nombre des items
  Integer lki = localKnownItems();
  // ReduceSum sur l'ensemble de la topologie
  Integer gki = m_kernel->parallel()->reduce(Parallel::ReduceSum, lki);
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] Working with lki="
          << lki << ", gki=" << gki << "\33[m";
  // Initialisation du kernel d'Aleph en fonction les locals et globals known items
  m_kernel->initialize(gki, lki);
  // A partir d'ici, le kernel est initialisé, on utilise directement les m_arguments_queue
  // LA topologie a été remplacée par une nouvelle
  debug() << "\33[1;7;33m[AlephIndexing::nowYouCanBuildTheTopology] Kernel is now initialized, rewinding Aleph operations!\33[m";
  // Si on est en parallèle, il faut consolider les indices suivant la nouvelle topologie
  if (m_kernel->isParallel())
    buildIndexesFromAddress();
  // On peut maintenant créer le triplet (matrice,lhs,rhs)
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
  // Et on revient pour rejouer les setValues de la matrice avec les indices consolidés
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] reSetValues fromThisMatrix\33[m";
  fromThisMatrix->reSetValuesIn(firstMatrix,
                                m_known_items_all_address);
  // Et on revient pour rejouer les addValues de la matrice avec les indices consolidés
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] reAddValues fromThisMatrix\33[m";
  fromThisMatrix->reAddValuesIn(firstMatrix,
                                m_known_items_all_address);
  // On reprovoque l'assemblage
  debug() << "\33[1;33m[AlephIndexing::nowYouCanBuildTheTopology] firstMatrix->assemble()\33[m";
  firstMatrix->assemble();
  // Et on fait le même processus pour les lhs et rhs
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
