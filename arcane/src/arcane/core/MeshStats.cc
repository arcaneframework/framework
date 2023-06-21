// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshStats.cc                                                (C) 2000-2023 */
/*                                                                           */
/* Statistiques sur le maillage.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Collection.h"

#include "arcane/core/MeshStats.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/StringDictionary.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshStats* IMeshStats::
create(ITraceMng* trace,IMesh* mesh,IParallelMng* pm)
{
  return new MeshStats(trace,mesh,pm);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshStats::
MeshStats(ITraceMng* trace,IMesh* mesh,IParallelMng* pm)
: TraceAccessor(trace)
, m_mesh(mesh)
, m_parallel_mng(pm)
, m_dictionary(new StringDictionary())
{
  m_dictionary->add(String("Cell"),String("Maille"));
  m_dictionary->add(String("Node"),String("Noeud"));
  m_dictionary->add(String("Face"),String("Face"));
  m_dictionary->add(String("Edge"),String("Arête"));
  m_dictionary->add(String("Triangle3"),String("Triangle3"));
  m_dictionary->add(String("Quad4"),String("Quadrangle4"));
  m_dictionary->add(String("Pentagon5"),String("Pentagone5"));
  m_dictionary->add(String("Hexagon6"),String("Hexagone6"));
  m_dictionary->add(String("Pyramid"),String("Pyramide"));
  m_dictionary->add(String("Hexaedron"),String("Hexaèdre"));
  m_dictionary->add(String("Pentaedron"),String("Pentaèdre"));
  m_dictionary->add(String("Tetraedron"),String("Tetraèdre"));
  m_dictionary->add(String("Wedge7"),String("Prisme7"));
  m_dictionary->add(String("Wedge8"),String("Prisme8"));
  m_dictionary->add(String("Link"),String("Liaison"));
  m_dictionary->add(String("DualNode"),String("Noeud Dual"));
  m_dictionary->add(String("DualEdge"),String("Arête Duale"));
  m_dictionary->add(String("DualFace"),String("Face Duale"));
  m_dictionary->add(String("DualCell"),String("Maille Duale"));
  // Les traductions des éléments non explicitement traduits sont à l'identique
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshStats::
~MeshStats()
{
  delete m_dictionary;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::dumpStats()
{
  // Affichage du maillage
  _dumpStats<IMesh>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::dumpGraphStats()
{
  _dumpStats<IMesh>();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> void MeshStats::
_computeElementsOnGroup<IMesh>(Int64ArrayView nb_type,Int64ArrayView nb_kind, Integer istat)
{
  _computeElementsOnGroup(nb_type,nb_kind,m_mesh->allNodes(),istat);
  _computeElementsOnGroup(nb_type,nb_kind,m_mesh->allEdges(),istat);
  _computeElementsOnGroup(nb_type,nb_kind,m_mesh->allFaces(),istat);
  _computeElementsOnGroup(nb_type,nb_kind,m_mesh->allCells(),istat);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> void MeshStats::
_statLabel<IMesh>(String name)
{
  info() << " -- MESH STATISTICS " << Trace::Width(8) << name 
         << "    FOR " << m_mesh->name();
}
  

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void MeshStats::
_dumpStats()
{
  Trace::Setter setter(traceMng(),"Mesh");

  const Integer nb_type = ItemTypeMng::nbBasicItemType();
  const Integer nb_kind = NB_ITEM_KIND;

  const char* name[3] = { "(All)", "(Own)", "(Ghost)" };
  
  // Pour ne s'occuper de Own et Ghost que en parallèle
  const Integer nstat = (m_parallel_mng->isParallel()) ? 3 : 1;

  // On regroupe toutes les données dans la même communication
  const Integer data_by_stat = nb_type+nb_kind;
  const Integer all_data_count = nstat*data_by_stat;
  Int64UniqueArray nb_data(all_data_count,0);
  
  // On remplit maintenant tout le tableau par bloc
  for( Integer istat=0; istat<nstat; ++istat){
    const Integer first_type = istat * data_by_stat;
    const Integer first_kind = first_type + nb_type;
    // Nombre d'éléments de chaque type sur ce sous-domaine.
    Int64ArrayView nb_local_type = nb_data.view().subView(first_type,nb_type);
    // Nombre d'éléments de chaque genre sur de sous-domaine
    Int64ArrayView nb_local_kind = nb_data.view().subView(first_kind,nb_kind);
    _computeElementsOnGroup<T>(nb_local_type,nb_local_kind,istat);
  }  

  // Tableau résultat de synthèse
  Int64UniqueArray nb_data_min(all_data_count,0);
  Int64UniqueArray nb_data_max(all_data_count,0);
  Int64UniqueArray nb_data_sum(all_data_count,0);
  Int32UniqueArray min_data_rank(all_data_count,0);
  Int32UniqueArray max_data_rank(all_data_count,0);
  m_parallel_mng->computeMinMaxSum(nb_data,
                                   nb_data_min,
                                   nb_data_max,
                                   nb_data_sum,
                                   min_data_rank,
                                   max_data_rank);

  // On vérifie s'il y a qqchose à afficher
  {
    bool is_empty = true;
    for( Integer istat=0; istat<nstat; ++istat){
      const Integer first_type = istat * data_by_stat;
      Int64ConstArrayView nb_global_type = nb_data_sum.subConstView(first_type,nb_type);
      for(Integer i = 0; i < nb_global_type.size(); ++i)
        if(nb_global_type[i] != 0) {
          is_empty = false;
          break;
        }
      if (!is_empty)
        break;
    }
    if (is_empty)
      return;
  }

  ItemTypeMng* type_mng = m_mesh->itemTypeMng();

  // On produit maintenant l'affichage des stats
  const Integer nb_rank = m_parallel_mng->commSize();
  for( Integer istat=0; istat<nstat; ++istat){
    // Construction des vues de travail décompactées 
    // (on reprend les noms des variables de la précédente version)
    const Integer first_type = istat * data_by_stat;
    const Integer first_kind = first_type + nb_type;

    Int64ConstArrayView nb_local_type     = nb_data.subConstView(first_type,nb_type);
    Int64ConstArrayView nb_local_min_type = nb_data_min.subConstView(first_type,nb_type);
    Int64ConstArrayView nb_local_max_type = nb_data_max.subConstView(first_type,nb_type);
    Int64ConstArrayView nb_global_type    = nb_data_sum.subConstView(first_type,nb_type);
    Int32ConstArrayView min_rank_type     = min_data_rank.subConstView(first_type,nb_type);
    Int32ConstArrayView max_rank_type     = max_data_rank.subConstView(first_type,nb_type);

    Int64ConstArrayView nb_local_kind     = nb_data.subConstView(first_kind,nb_kind);
    Int64ConstArrayView nb_local_min_kind = nb_data_min.subConstView(first_kind,nb_kind);
    Int64ConstArrayView nb_local_max_kind = nb_data_max.subConstView(first_kind,nb_kind);
    Int64ConstArrayView nb_global_kind    = nb_data_sum.subConstView(first_kind,nb_kind);
    Int32ConstArrayView min_rank_kind     = min_data_rank.subConstView(first_kind,nb_kind);
    Int32ConstArrayView max_rank_kind     = max_data_rank.subConstView(first_kind,nb_kind);

    info() << " -------------------------------------------";
    _statLabel<T>(name[istat]);

    info() << Trace::Width(18) << "Item"
           << Trace::Width(10) << "Myself"
           << Trace::Width(10) << "Min"
           << Trace::Width(8) << "Rank"
           << Trace::Width(10) << "Max"
           << Trace::Width(8) << "Rank"
           << Trace::Width(10) << "Average"
           << Trace::Width(10) << "Bal"
           << Trace::Width(12) << "Total";

    info() << " ";
    for( Integer i=0, s=nb_kind; i<s; ++i ){
      eItemKind kt = static_cast<eItemKind>(i);
      _printInfo(itemKindName(kt),nb_local_kind[i],
                 nb_local_min_kind[i],min_rank_kind[i],
                 nb_local_max_kind[i],max_rank_kind[i],
                 nb_global_kind[i],nb_rank);
      // if (nb_global_kind[i]!=0) {
      //   pinfo() << "MeshStats ("<<  m_parallel_mng->commRank() <<") : " << name[istat]
      //         << " kind=" << itemKindName(kt)
      //         << " n=" << nb_local_kind[i];
      //   plog() << "MeshStats ("<<  m_parallel_mng->commRank() <<") : " << name[istat]
      //         << " kind=" << itemKindName(kt)
      //         << " n=" << nb_local_kind[i];
      // }
    }
    info() << " ";
    for( Integer i=0, s=nb_type; i<s; ++i ) {
      _printInfo(type_mng->typeName(i),nb_local_type[i],
                 nb_local_min_type[i],min_rank_type[i],
                 nb_local_max_type[i],max_rank_type[i],
                 nb_global_type[i],nb_rank);
      // if (nb_global_type[i]!=0)
      //   pinfo() << "MeshStats ("<<  m_parallel_mng->commRank() <<") : " << name[istat]
      //         << " type=" << Item::typeName(i)
      //         << " n=" << nb_local_type[i];

    }
    info() << " ";
    info() << " -------------------------------------------";
  }
  // _computeNeighboorsComm();

  _dumpCommunicatingRanks();
  _dumpLegacyConnectivityMemoryUsage();
  _dumpIncrementalConnectivityMemoryUsage();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::
_dumpLegacyConnectivityMemoryUsage()
{
  // Cette liste de nom est issue de mesh/ItemFamily.cc
  UniqueArray<String> var_names = { "FamilyItemsData",  "FamilyItemsShared" };
  IVariableMng* vm = m_mesh->variableMng();
  Real total_memory = 0.0;
  for( IItemFamily* family : m_mesh->itemFamilies() ){
    String name = family->name();
    Real family_memory = 0.0;
    info(4) << "Family name=" << family->name();
    for( String s : var_names ){
      IVariable* v = vm->findMeshVariable(m_mesh,name+s);
      if (v){
        Real v_memory = v->allocatedMemory();
        family_memory += v_memory;
        info(4) << "Allocated Memory n=" << s << " v=" << v_memory;
      }
    }
    info(4) << "Memory for family name=" << name << " mem=" << family_memory;
    total_memory += family_memory;
  }
  info() << "Total memory for legacy connectivities mem=" << total_memory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::
_dumpIncrementalConnectivityMemoryUsage()
{
  IVariableMng* vm = m_mesh->variableMng();
  Real total_memory = 0.0;
  VariableCollection used_variables = vm->usedVariables();
  const String tag_name = "ArcaneConnectivity";
  for( VariableCollection::Enumerator iv(used_variables); ++iv; ){
    IVariable* v = *iv;
    if (!v->hasTag(tag_name))
      continue;
    if (v->meshHandle().meshOrNull()==m_mesh){
      Real v_memory = v->allocatedMemory();
      info(4) << "Allocated Memory n=" << v->name() << " v=" << v_memory;
      total_memory += v_memory;
    }
  }
  info() << "Total memory for incremental connectivities mem=" << total_memory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::
_printInfo(const String& name,Int64 nb_local,
           Int64 nb_local_min,Integer min_rank,
           Int64 nb_local_max,Integer max_rank,
           Int64 nb_global,Integer nb_rank)
{
  if (nb_global==0 && nb_local==0)
    return;

  String tr_name = m_dictionary->find(String(name));
  if (tr_name.empty()) tr_name = name;

  Int64 average = nb_global;
  if (nb_rank!=0)
    average /= nb_rank;
  Real bal1 = (Real)(nb_local_max - average);
  if (average!=0)
    bal1 = bal1 / (Real)average;
  Int64 bal = (Int64)(bal1*1000);
  info() << Trace::Width(18) << tr_name
         << Trace::Width(10) << nb_local
         << Trace::Width(10) << nb_local_min
         << Trace::Width(8) << min_rank
         << Trace::Width(10) << nb_local_max
         << Trace::Width(8) << max_rank
         << Trace::Width(10) << average
         << Trace::Width(10) << bal
         << Trace::Width(12) << nb_global;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::
_computeElementsOnGroup(Int64ArrayView nb_type,Int64ArrayView nb_kind,
                        ItemGroup group,Integer istat)
{
  Integer ik = (Integer)group.itemKind();
  ENUMERATE_ITEM(i,group){
    Item item = *i;
    int type = item.type();
    if (istat==0 || (istat==1 && item.isOwn()) || (istat==2 && !item.isOwn())){
      ++nb_kind[ik];
      ++nb_type[type];
    }
  }
}

void MeshStats::
_computeNeighboorsComm()
{
  Integer nb_proc = m_parallel_mng->commSize();
  Int64UniqueArray vol_comm_out(nb_proc);
  StringBuilder out = "";
  vol_comm_out.fill(0);

  ENUMERATE_ITEM(i,m_mesh->allCells()){
    Item item = *i;
    if (item.isOwn())
      continue;
    vol_comm_out[item.owner()]++;
  }

  for (Integer i =0 ; i < nb_proc ; ++i) {
    out += " ";
    out += vol_comm_out[i];
  }
  pinfo() << "Comm Proc " << m_parallel_mng->commRank() << " : " << out;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshStats::
_dumpCommunicatingRanks()
{
  // Affiche les statistiques sur le nombre de sous-domaines avec
  // lequels on communique. On prend les voisins sur la liste des mailles.
  IItemFamily* cell_family = m_mesh->cellFamily();
  IVariableSynchronizer* sync_info = cell_family->allItemsSynchronizer();
  Int64 nb_comm_local = sync_info->communicatingRanks().size();
  Int32 nb_comm_min_rank = 0;
  Int32 nb_comm_max_rank = 0;
  Int64 nb_comm_max = 0;
  Int64 nb_comm_min = 0;
  Int64 nb_comm_sum = 0;
  IParallelMng* pm = m_parallel_mng;
  pm->computeMinMaxSum(nb_comm_local,nb_comm_min,nb_comm_max,nb_comm_sum,
                       nb_comm_min_rank,nb_comm_max_rank);
  Int64 average = nb_comm_sum / pm->commSize();
  info() << String::format("CommunicatingSubDomains: local={0}, min={1}, max={2}"
                           " average={3} min_rank={4} max_rank={5}",
                           nb_comm_local,nb_comm_min,nb_comm_max,average,
                           nb_comm_min_rank,nb_comm_max_rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namepsace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
