// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfoList.cc                                       (C) 2000-2022 */
/*                                                                           */
/* Liste de 'ItemSharedInfo'.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/VariableTypes.h"

#include "arcane/mesh/ItemSharedInfoList.h"
#include "arcane/mesh/ItemFamily.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemSharedInfoList::ItemNumElements
{
 public:

  explicit ItemNumElements(Integer type)
  : m_type(type) {}

 public:

  static bool m_debug;

  Integer m_type;

  bool operator<(const ItemNumElements& b) const
    {
#ifdef ARCANE_CHECK
      if (m_debug){
        cout << "Compare:\nTHIS=";
        print(cout);
        cout << "\nTO=";
        b.print(cout);
        cout << "\nVAL=" << compare(b);
        cout << "\n";
      }
#endif
      return compare(b);
    }
 private:
  inline bool compare(const ItemNumElements& b) const
  {
    return m_type<b.m_type;
  }
 public:
  void print(std::ostream& o) const
  {
    o << " Type=" << m_type;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemSharedInfoList::ItemNumElements::m_debug = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o,const ItemSharedInfoList::ItemNumElements& v)
{
  v.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ItemSharedInfoList::Variables
{
 public:
  Variables(IMesh* mesh,const String& name)
  : m_infos_values(VariableBuildInfo(mesh,name)){}
 public:
  VariableArray2Int32 m_infos_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfoList::
ItemSharedInfoList(ItemFamily* family)
: TraceAccessor(family->traceMng())
, m_family(family)
, m_item_kind(family->itemKind())
, m_item_shared_infos_buffer(new MultiBufferT<ItemSharedInfoWithType>(100))
, m_infos_map(new ItemSharedInfoMap())
{
  {
    String var_name(family->name());
    var_name = var_name + "_SharedInfoList";
    m_variables = new Variables(family->mesh(),var_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfoList::
~ItemSharedInfoList()
{
  delete m_variables;
  delete m_infos_map;
  delete m_item_shared_infos_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
prepareForDump()
{
  Integer n = m_item_shared_infos.size();
  info(4) << "ItemSharedInfoList: write: " << m_family->name()
          << " count=" << n << " changed=" << m_list_changed;
  log() << "ItemSharedInfoList: write: " << m_family->name()
        << " count=" << n << " changed=" << m_list_changed;

  if (!m_list_changed)
    return;
  m_list_changed = false;
  //Integer n = m_item_shared_infos.size();
  Integer element_size = ItemSharedInfo::serializeWriteSize();
  m_variables->m_infos_values.resize(n,element_size);
  for( Integer i=0; i<n; ++i ){
    m_item_shared_infos[i]->serializeWrite(m_variables->m_infos_values[i]);
    //     if (i<20 && m_family->itemKind()==IK_Particle){
    //       ItemSharedInfo* isi = m_item_shared_infos[i];
    //       log() << "FAMILY" << m_family->name() << " ISI: i=" << i << " values=" << *isi;
    //       info() << "FAMILY" << m_family->name() << "ISI: i=" << i << " values=" << *isi;
    //     }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
readFromDump()
{
  Integer element_size = ItemSharedInfo::serializeSize();
  Integer n = m_variables->m_infos_values.dim1Size();
  info() << "ItemSharedInfoList: read: " << m_family->name() << " count=" << n;

  if (n>0){
    // Le nombre d'éléments sauvés dépend de la version de Arcane et du fait
    // qu'on utilise ou pas l'AMR.
    Integer stored_size = m_variables->m_infos_values[0].size();
    if (stored_size==ItemSharedInfo::serializeAMRSize()){
    }
    else if (stored_size!=element_size){
      // On ne peut relire que les anciennes versions (avant la 3.6)
      // dont la taille vaut 13 (avec AMR) ou 9 (sans AMR) ce qui correspond
      // aux versions de Arcane de 2021.
      if (stored_size!=13 && stored_size!=9)
        ARCANE_FATAL("Incoherence of saved data (most probably due to a"
                     " difference of versions between the protection and the executable."
                     " stored_size={0} element_size={1} count={2}",
                     stored_size,element_size,n);
    }
  }

  // Tous les types vont a nouveau être ajoutés à la liste
  m_item_shared_infos.clear();
  m_infos_map->clear();

  if (n==0)
    return;

  for( Integer i=0; i<n; ++i )
    allocOne();

  MeshItemInternalList* miil = m_family->mesh()->meshItemInternalList();
  ItemInternalConnectivityList* iicl = m_family->itemInternalConnectivityList();

  ItemTypeMng* itm = m_family->mesh()->itemTypeMng();
  for( Integer i=0; i<n; ++i ){
    Int32ConstArrayView buffer(m_variables->m_infos_values[i]);
    // Le premier élément du tampon contient toujours le type de l'entité
    ItemTypeInfo* it = itm->typeFromId(buffer[0]);
    ItemSharedInfoWithType* isi = m_item_shared_infos[i];
    *isi = ItemSharedInfoWithType(m_family,it,miil,iicl,m_family->viewsForItemSharedInfo(),buffer);

    ItemNumElements ine(it->typeId());
    std::pair<ItemSharedInfoMap::iterator,bool> old = m_infos_map->insert(std::make_pair(ine,isi));
    if (!old.second){
      // Vérifie que l'instance ajoutée ne remplace pas une instance déjà présente,
      // auquel il s'agit d'une erreur interne (opérateur de comparaison foireux)
      dumpSharedInfos();
      ItemNumElements::m_debug = true;
      bool compare = m_infos_map->find(ine)!=m_infos_map->end();
      fatal() << "INTERNAL: ItemSharedInfoList::readfromDump(): SharedInfo already present family=" << m_family->name()
              << "\nWanted:"
              << " type=" << it->typeId()
              << " compare=" << compare
              << "\nNEW_INE=(" << ine << ")"
              << "\nOLD_INE=(" << old.first->first << ")"
              << "\nNEW_ISI=(" << *isi << ")"
              << "\nOLD_ISI=(" << *old.first->second << ")";
     }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
checkValid()
{
  bool has_error = false;

  info() << "ItemSharedInfoList: check valid family=" << m_family->name()
         << " count=" << m_nb_item_shared_info
         << " list=" << m_item_shared_infos.size()
         << " free=" << m_free_item_shared_infos.size()
         << " changed=" << m_list_changed;

  // Premièrement, le item->localId() doit correspondre à l'indice
  // dans le tableau m_internal
  for( Integer i=0, is=m_item_shared_infos.size(); i<is; ++i ){
    ItemSharedInfo* item = m_item_shared_infos[i];
    if (item->index()!=i){
      error() << "The index (" << item->index() << ") from the list 'ItemSharedInfo' "
              << "of the family " << m_family->name() << " is not "
              << "coherent with its internal value (" << i << ")";
      has_error = true;
    }
  }
  if (has_error)
    ARCANE_FATAL("Internal error with the mesh structure");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemSharedInfoWithType* ItemSharedInfoList::
findSharedInfo(ItemTypeInfo* type)
{
  ItemNumElements ine(type->typeId());
  ItemSharedInfoMap::const_iterator i = m_infos_map->find(ine);
  if (i!=m_infos_map->end())
    return i->second;
  MeshItemInternalList* miil = m_family->mesh()->meshItemInternalList();
  ItemInternalConnectivityList* iicl = m_family->itemInternalConnectivityList();
  // Infos pas trouvé. On en construit une nouvelle
  ItemSharedInfoWithType* isi = allocOne();
  Integer old_index = isi->index();
  *isi = ItemSharedInfoWithType(m_family,type,miil,iicl,m_family->viewsForItemSharedInfo());
  isi->setIndex(old_index);
  std::pair<ItemSharedInfoMap::iterator,bool> old = m_infos_map->insert(std::make_pair(ine,isi));

  //#ifdef ARCANE_CHECK
  if (!old.second){
    // Vérifie que l'instance ajoutée ne remplace pas une instance déjà présente,
    // auquel il s'agit d'une erreur interne (opérateur de comparaison foireux)
    dumpSharedInfos();
    ItemNumElements::m_debug = true;
    bool compare = m_infos_map->find(ine)!=m_infos_map->end();
    fatal() << "INTERNE: ItemSharedInfoList::findSharedInfo() SharedInfo déjà présent\n"
            << "\nWanted:"
            << " type=" << type->typeId()
            << " compare=" << compare
            << "\nNEW_INE=(" << ine << ")"
            << "\nOLD_INE=(" << old.first->first << ")"
            << "\nNEW_ISI=(" << *isi << ")"
            << "\nOLD_ISI=(" << *old.first->second << ")";
  }
  //#endif
  return isi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoList::
dumpSharedInfos()
{
  info() << "--- ItemSharedInfos: family=" << m_family->name();
  for( ConstIterT<ItemSharedInfoMap> i(*m_infos_map); i(); ++i ){
    info() << "INE: " << i->first;
    info() << "ISI: " << *i->second;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
