// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupInternal.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de ItemGroup.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/ItemGroupInternal.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayUtils.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/ItemGroupObserver.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/MeshPartInfo.h"
#include "arcane/core/VariableUtils.h"
#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/ItemGroupImplInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
ItemGroupInternal()
: m_internal_api(this)
, m_sub_parts_by_type(this)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
ItemGroupInternal(IItemFamily* family,const String& name)
: m_internal_api(this)
, m_mesh(family->mesh())
, m_item_family(family)
, m_variable_name(String("GROUP_")+family->name()+name)
, m_is_null(false)
, m_kind(family->itemKind())
, m_name(name)
, m_sub_parts_by_type(this)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
ItemGroupInternal(IItemFamily* family,ItemGroupImpl* parent,const String& name)
: m_internal_api(this)
, m_mesh(parent->mesh())
, m_item_family(family)
, m_parent(parent)
, m_variable_name(String("GROUP_")+m_item_family->name()+name)
, m_is_null(false)
, m_kind(family->itemKind())
, m_name(name)
, m_sub_parts_by_type(this)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
~ItemGroupInternal()
{
  // (HP) TODO: vérifier qu'il n'y a plus d'observer à cet instant
  // Ceux des sous-groupes n'ont pas été détruits
  for( const auto& i : m_observers ) {
    delete i.second;
  }
  delete m_variable_items_local_id;
  delete m_compute_functor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
_init()
{
  if (m_item_family)
    m_full_name = m_item_family->fullName() + "_" + m_name;

  // Si un maillage est associé et qu'on n'est un groupe enfant alors les données du groupe
  // sont conservées dans une variable.
  if (m_mesh && !m_parent){
    int property = IVariable::PSubDomainDepend | IVariable::PPrivate;
    VariableBuildInfo vbi(m_mesh,m_variable_name,property);
    m_variable_items_local_id = new VariableArrayInt32(vbi);
    m_items_local_id = &m_variable_items_local_id->_internalTrueData()->_internalDeprecatedValue();
    updateTimestamp();
  }

  // Regarde si on utilise la version 2 pour ApplyOperationByBasicType
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_APPLYOPERATION_VERSION", true))
    m_sub_parts_by_type.m_use_v2_for_apply_operation = (v.value() == 2);

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DEBUG_APPLYOPERATION", true))
    m_sub_parts_by_type.m_is_debug_apply_operation = (v.value() > 0);

  m_is_check_simd_padding = arcaneIsCheck();
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CHECK_SIMDPADDING", true)){
    m_is_check_simd_padding = (v.value()>0);
    m_is_print_check_simd_padding = (v.value()>1);
  }

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_PRINT_APPLYSIMDPADDING", true)){
    m_is_print_apply_simd_padding = (v.value()>0);
    m_is_print_stack_apply_simd_padding = (v.value()>1);
  }

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_LOCK_FOR_ITEMGROUP_UPDATE", true)) {
    if (v.value() > 0)
      m_check_need_update_mutex.create();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalList ItemGroupInternal::
items() const
{
  if (m_item_family)
    return m_item_family->itemsInternal();
  return m_mesh->itemsInternal(m_kind);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemGroupInternal::
maxLocalId() const
{
  return m_item_family->maxLocalId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInfoListView ItemGroupInternal::
itemInfoListView() const
{
  if (m_item_family)
    return m_item_family->itemInfoListView();
  return m_mesh->itemFamily(m_kind)->itemInfoListView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
resetSubGroups()
{
  if (!m_is_all_items)
    ARCANE_FATAL("Call to _resetSubGroups() is only valid for group of AllItems");

  m_own_group = nullptr;
  m_ghost_group = nullptr;
  m_interface_group = nullptr;
  m_node_group = nullptr;
  m_edge_group = nullptr;
  m_face_group = nullptr;
  m_cell_group = nullptr;
  m_inner_face_group = nullptr;
  m_outer_face_group = nullptr;
  m_active_cell_group = nullptr;
  m_own_active_cell_group = nullptr;
  m_active_face_group = nullptr;
  m_own_active_face_group = nullptr;
  m_inner_active_face_group = nullptr;
  m_outer_active_face_group = nullptr;
  m_level_cell_group.clear();
  m_own_level_cell_group.clear();
  m_sub_parts_by_type.clear();
  m_sub_groups.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
notifyExtendObservers(const Int32ConstArrayView * info)
{
  ARCANE_ASSERT((!m_need_recompute || m_is_all_items),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeExtend(info);
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
notifyReduceObservers(const Int32ConstArrayView * info)
{
  ARCANE_ASSERT((!m_need_recompute || m_is_all_items),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeReduce(info);
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
notifyCompactObservers(const Int32ConstArrayView * info)
{
  ARCANE_ASSERT((!m_need_recompute || m_is_all_items),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeCompact(info);
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->compact(info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
notifyInvalidateObservers()
{
#ifndef NO_USER_WARNING
#warning "(HP) Assertion need fix"
#endif /* NO_USER_WARNING */
  // Cela peut se produire en cas d'invalidation en cascade
  // ARCANE_ASSERT((!m_need_recompute),("Operation on invalid group"));
  for( const auto& i : m_observers ) {
    IItemGroupObserver * obs = i.second;
    obs->executeInvalidate();
  }
  if (m_group_index_table.isUsed())
    m_group_index_table->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que les localIds() sont contigüs.
 */
void ItemGroupInternal::
checkIsContiguous()
{
  m_is_contiguous = false;
  Int32ConstArrayView lids = itemsLocalId();
  if (lids.empty()) {
    m_is_contiguous = false;
    return;
  }
  Int32 first_lid = lids[0];

  bool is_bad = false;
  for( Integer i=0, n=lids.size(); i<n; ++i ){
    if (lids[i]!=(first_lid+i)){
      is_bad = true;
      break;
    }
  }
  if (!is_bad)
    m_is_contiguous = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
applySimdPadding()
{
  if (m_is_print_apply_simd_padding){
    String stack;
    if (m_is_print_stack_apply_simd_padding)
      stack = String(" stack=") + platform::getStackTrace();
    ITraceMng* tm = m_item_family->traceMng();
    tm->info() << "ApplySimdPadding group_name=" << m_name << stack;
  }
  // Fait un padding des derniers éléments du tableau en recopiant la
  // dernière valeurs.
  m_internal_api.notifySimdPaddingDone();
  Arcane::applySimdPadding(mutableItemsLocalId());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les derniers éléments du groupe pour avoir un vecteur
 * SIMD complet.
 *
 * Pour que la vectorisation fonctionne il faut que le nombre d'éléments
 * du groupe soit un multiple de la taille d'un vecteur SIMD. Si ce n'est
 * pas le cas, on remplit les dernières valeurs du tableau des localId()
 * avec le dernier élément.
 *
 * Par exemple, on supporse une taille d'un vecteur SIMD de 8 (ce qui est le maximum
 * actuellement avec l'AVX512) et un groupe \a grp de 13 éléments. Il faut donc
 * remplit le groupe comme suit:
 * \code
 * Int32 last_local_id = grp[12];
 * grp[13] = grp[14] = grp[15] = last_local_id.
 * \endcode
 *
 * A noter que la taille du groupe reste effectivement de 13 éléments. Le
 * padding supplémentaire n'est que pour les itérations via ENUMERATE_SIMD.
 * Comme le tableau des localId() est alloué avec l'allocateur d'alignement
 * il est garanti que la mémoire allouée est suffisante pour faire le padding.
 *
 * \todo Ne pas faire cela dans tous les checkNeedUpdate() mais mettre
 * en place une méthode qui retourne un énumérateur spécifique pour
 * la vectorisation.
 */
void ItemGroupInternal::
checkUpdateSimdPadding()
{
  if (m_simd_timestamp >= timestamp()){
    // Vérifie que le padding est bon
    if (m_is_check_simd_padding){
      if (m_is_print_check_simd_padding && m_item_family){
        ITraceMng* tm = m_item_family->traceMng();
        tm->info() << "check padding name=" << fullName()
                   << " timestamp=" << timestamp()
                   << " simd_timestamp=" << m_simd_timestamp
                   << " size=" << mutableItemsLocalId().size()
                   << " capacity=" << mutableItemsLocalId().capacity();
      }
      ArrayUtils::checkSimdPadding(itemsLocalId());
    }
    return;
  }
  this->applySimdPadding();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
_removeItems(SmallSpan<const Int32> items_local_id)
{
  if ( !((!m_need_recompute && !isAllItems()) || (m_transaction_mode && isAllItems())) )
    ARCANE_FATAL("Operation on invalid group");

  if (m_compute_functor && !m_transaction_mode)
    ARCANE_FATAL("Cannot remove items on computed group ({0})", name());

  IMesh* amesh = mesh();
  if (!amesh)
    throw ArgumentException(A_FUNCINFO,"null group");

  ITraceMng* trace = amesh->traceMng();
  if (isOwn() && amesh->meshPartInfo().nbPart()!=1)
    ARCANE_THROW(NotSupportedException,"Cannot remove items if isOwn() is true");

  Int32 nb_item_to_remove = items_local_id.size();

  // N'est utile que si on a des observers
  UniqueArray<Int32> removed_local_ids;

  if (nb_item_to_remove!=0) { // on ne peut tout de même pas faire de retour anticipé à cause des observers

    Int32Array& items_lid = mutableItemsLocalId();
    const Int32 old_size = items_lid.size();
    bool has_removed = false;
   
    if (isAllItems()) {
      // Algorithme anciennement dans DynamicMeshKindInfo
      // Supprime des items du groupe all_items par commutation avec les 
      // éléments de fin de ce groupe
      // ie memoire persistante O(size groupe), algo O(remove items)
      has_removed = true;
      Integer nb_item = old_size;
      for( Integer i=0, n=nb_item_to_remove; i<n; ++i ){
        Int32 removed_local_id = items_local_id[i];
        Int32 index = m_items_index_in_all_group[removed_local_id];
        --nb_item;
        Int32 moved_local_id = items_lid[nb_item];
        items_lid[index] = moved_local_id;
        m_items_index_in_all_group[moved_local_id] = index;
      }
      items_lid.resize(nb_item);
    }
    else {
      // Algorithme pour les autres groupes
      // Décalage de tableau
      // ie mémoire locale O(size groupe), algo O(size groupe)
      // Marque un tableau indiquant les entités à supprimer
      UniqueArray<bool> remove_flags(maxLocalId(),false);
      for( Int32 i=0; i<nb_item_to_remove; ++i )
        remove_flags[items_local_id[i]] = true;
    
      {
        Int32 next_index = 0;
        for( Int32 i=0; i<old_size; ++i ){
          Int32 lid = items_lid[i];
          if (remove_flags[lid]) {
            removed_local_ids.add(lid);
            continue;
          }
          items_lid[next_index] = lid;
          ++next_index;
        }
        if (next_index!=old_size) {
          has_removed = true;
          items_lid.resize(next_index);
        }
      }
    }
  
    updateTimestamp();
    if (arcaneIsCheck()){
      trace->info(5) << "ItemGroupImpl::removeItems() group <" << name() << "> "
                     << " old_size=" << old_size
                     << " new_size=" << nbItem()
                     << " removed?=" << has_removed;
      checkValid();
    }
  }

  Int32ConstArrayView observation_info(removed_local_ids);
  notifyReduceObservers(&observation_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
checkValid()
{
  ITraceMng* msg = mesh()->traceMng();
  if (m_need_recompute && m_compute_functor) {
    msg->debug(Trace::High) << "ItemGroupImpl::checkValid on " << name() << " : skip group to recompute";
    return;
  }

  // Les points suivants sont vérifiés:
  // - une entité n'est présente qu'une fois dans le groupe
  // - les entités du groupe ne sont pas détruites
  UniqueArray<bool> presence_checks(maxLocalId());
  presence_checks.fill(0);
  Integer nb_error = 0;

  ItemInternalList items(this->items());
  Integer items_size = items.size();
  Int32ConstArrayView items_lid(itemsLocalId());

  for( Integer i=0, is=items_lid.size(); i<is; ++i ){
    Integer lid = items_lid[i];
    if (lid>=items_size){
      if (nb_error<10){
        msg->error() << "Wrong local index lid=" << lid << " max=" << items_size
                     << " var_max_size=" << maxLocalId();
      }
      ++nb_error;
      continue;
    }
    ItemInternal* item = items[lid];
    if (item->isSuppressed()){
      if (nb_error<10){
        msg->error() << "Item " << ItemPrinter(item) << " in group "
                     << name() << " does not exist anymore";
      }
      ++nb_error;
    }
    if (presence_checks[lid]){
      if (nb_error<10){
        msg->error() << "Item " << ItemPrinter(item) << " in group "
                     << name() << " was found twice or more";
      }
      ++nb_error;
    }
    presence_checks[lid] = true;
  }
  if (isAllItems()) {
    for( Integer i=0, n=items_lid.size(); i<n; ++i ){
      Int32 local_id = items_lid[i];
      Int32 index_in_all_group = m_items_index_in_all_group[local_id];
      if (index_in_all_group!=i){
        if (nb_error<10){
          msg->error() << A_FUNCINFO
                       << ": " << itemKindName(m_kind)
                       << ": incoherence between 'local_id' and index in the group 'All' "
                       << " i=" << i
                       << " local_id=" << local_id
                       << " index=" << index_in_all_group;
        }
        ++nb_error;
      }
    }
  }
  if (nb_error!=0) {
    String parent_name = "none";
    if (m_parent)
      parent_name = m_parent->name();
    ARCANE_FATAL("Error in group name='{0}' parent='{1}' nb_error={2}",
                 name(),parent_name,nb_error);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
_notifyDirectRemoveItems(SmallSpan<const Int32> removed_ids, Int32 nb_remaining)
{
  mutableItemsLocalId().resize(nb_remaining);
  updateTimestamp();
  if (arcaneIsCheck())
    checkValid();
  // NOTE: la liste \a removed_ids n'est pas forcément dans le même ordre
  // que celle obtenue via removeItems()
  Int32ConstArrayView observation_info(removed_ids.smallView());
  notifyReduceObservers(&observation_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplInternal::
setAsConstituentGroup()
{
  m_p->m_is_constituent_group = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SmallSpan<Int32> ItemGroupImplInternal::
itemsLocalId()
{
  return m_p->itemsLocalId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplInternal::
notifyDirectRemoveItems(SmallSpan<const Int32> removed_ids, Int32 nb_remaining)
{
  m_p->_notifyDirectRemoveItems(removed_ids, nb_remaining);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplInternal::
notifySimdPaddingDone()
{
  m_p->m_simd_timestamp = m_p->timestamp();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupImplInternal::
setMemoryRessourceForItemLocalId(eMemoryRessource mem)
{
  VariableArrayInt32* v = m_p->m_variable_items_local_id;
  if (v)
    VariableUtils::experimentalChangeAllocator(v->variable(),mem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
