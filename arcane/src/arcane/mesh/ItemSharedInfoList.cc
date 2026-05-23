// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemSharedInfoList.cc                                       (C) 2000-2022 */
/*                                                                           */
/* List of 'ItemSharedInfo'.                                                 */
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

ItemSharedInfoWithType::
ItemSharedInfoWithType(ItemSharedInfo* shared_info,ItemTypeInfo* item_type)
: m_shared_info(shared_info)
, m_type_id(item_type->typeId())
{
}

ItemSharedInfoWithType::
ItemSharedInfoWithType(ItemSharedInfo* shared_info,ItemTypeInfo* item_type,Int32ConstArrayView buffer)
: m_shared_info(shared_info)
, m_type_id(item_type->typeId())
{
  // The buffer size depends on the versions of Arcane.
  // Before 3.2 (October 2021), the buffer size is 9 (non-AMR) or 13 (AMR)
  // Between 3.2 and 3.6 (May 2022), the size is always 13
  // Starting from 3.6, the size is 6.
  //
  // We are not trying to recover with versions
  // of Arcane prior to 3.2, so we can assume that the size
  // of the buffer is 13. These versions use the new connectivity
  // and therefore the number of elements is always 0 (as well as the *allocated)
  // except for m_nb_node.
  //
  // Starting from 3.6, the number of nodes is no longer used
  // and is always 0. We can therefore delete these fields from ItemSharedInfo
  // for the end of 2022 versions.

  // TODO: Indicate that starting from version 3.7 we only support
  // buf_size==6 with version number 0x0307
  Int32 buf_size = buffer.size();
  if (buf_size!=6)
    ARCANE_FATAL("Invalid buf size '{0}'. This is probably because your checkpoint is from a version of Arcane which is too old (before 3.6)",
                 buf_size);
  m_index = buffer[2];
  m_nb_reference = buffer[3];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemSharedInfoWithType::
serializeWrite(Int32ArrayView buffer)
{
  buffer[0] = m_type_id; // Must always be the first

  buffer[1] = 0x0307; // Version number (3.7).

  buffer[2] = m_index;
  buffer[3] = m_nb_reference;

  buffer[4] = 0; // Reserved
  buffer[5] = 0; // Reserved
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
ItemSharedInfoList(ItemFamily* family,ItemSharedInfo* common_shared_info)
: TraceAccessor(family->traceMng())
, m_family(family)
, m_common_item_shared_info(common_shared_info)
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
  Integer element_size = ItemSharedInfoWithType::serializeSize();
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
  Integer element_size = ItemSharedInfoWithType::serializeSize();
  Integer n = m_variables->m_infos_values.dim1Size();
  info() << "ItemSharedInfoList: read: " << m_family->name() << " count=" << n;

  if (n>0){
    // The number of saved elements depends on the Arcane version and whether
    // AMR is used.
    Integer stored_size = m_variables->m_infos_values[0].size();
    if (stored_size==ItemSharedInfoWithType::serializeSize()){
    }
    else if (stored_size!=element_size){
      // We can only read older versions (before 3.6)
      // whose size is 13 (with AMR) or 9 (without AMR), which corresponds
      // to Arcane versions from 2021.
      if (stored_size!=13 && stored_size!=9)
        ARCANE_FATAL("Incoherence of saved data (most probably due to a"
                     " difference of versions between the protection and the executable."
                     " stored_size={0} element_size={1} count={2}",
                     stored_size,element_size,n);
    }
  }

  // All types will be added back to the list
  m_item_shared_infos.clear();
  m_infos_map->clear();

  if (n==0)
    return;

  for( Integer i=0; i<n; ++i )
    allocOne();

  ItemTypeMng* itm = m_family->mesh()->itemTypeMng();
  for( Integer i=0; i<n; ++i ){
    Int32ConstArrayView buffer(m_variables->m_infos_values[i]);
    // The first element of the buffer always contains the entity type
    ItemTypeInfo* it = itm->typeFromId(buffer[0]);
    ItemSharedInfoWithType* isi = m_item_shared_infos[i];
    *isi = ItemSharedInfoWithType(m_common_item_shared_info,it,buffer);

    ItemNumElements ine(it->typeId());
    std::pair<ItemSharedInfoMap::iterator,bool> old = m_infos_map->insert(std::make_pair(ine,isi));
    if (!old.second){
      // Check that the added instance does not replace an already present instance,
      // which is an internal error (faulty comparison operator)
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

  // Firstly, item->localId() must correspond to the index
  // in the m_internal array
  for( Integer i=0, n=m_item_shared_infos.size(); i<n; ++i ){
    ItemSharedInfoWithType* item = m_item_shared_infos[i];
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
  // Info not found. We build a new one
  ItemSharedInfoWithType* isi = allocOne();
  Integer old_index = isi->index();
  *isi = ItemSharedInfoWithType(m_common_item_shared_info,type);
  isi->setIndex(old_index);
  std::pair<ItemSharedInfoMap::iterator,bool> old = m_infos_map->insert(std::make_pair(ine,isi));

  //#ifdef ARCANE_CHECK
  if (!old.second){
    // Check that the added instance does not replace an already present instance,
    // which is an internal error (faulty comparison operator)
    dumpSharedInfos();
    ItemNumElements::m_debug = true;
    bool compare = m_infos_map->find(ine)!=m_infos_map->end();
    fatal() << "INTERNAL: ItemSharedInfoList::findSharedInfo() SharedInfo already present\n"
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
