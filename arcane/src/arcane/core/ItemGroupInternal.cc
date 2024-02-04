// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupInternal.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de ItemGroup.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/ItemGroupInternal.h"

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ArrayUtils.h"

#include "arcane/core/ItemGroupObserver.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/datatype/DataAllocationInfo.h"
#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
ItemGroupInternal()
: m_mesh(nullptr)
, m_item_family(nullptr)
, m_parent(nullptr)
, m_variable_name()
, m_is_null(true)
, m_kind(IK_Unknown)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
ItemGroupInternal(IItemFamily* family,const String& name)
: m_mesh(family->mesh())
, m_item_family(family)
, m_parent(nullptr)
, m_variable_name(String("GROUP_")+family->name()+name)
, m_is_null(false)
, m_kind(family->itemKind())
, m_name(name)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupInternal::
ItemGroupInternal(IItemFamily* family,ItemGroupImpl* parent,const String& name)
: m_mesh(parent->mesh())
, m_item_family(family)
, m_parent(parent)
, m_variable_name(String("GROUP_")+m_item_family->name()+name)
, m_is_null(false)
, m_kind(family->itemKind())
, m_name(name)
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
  // Si maillage associé et pas un groupe enfant alors les données du groupe
  // sont conservées dans une variable.
  if (m_mesh && !m_parent){
    int property = IVariable::PSubDomainDepend | IVariable::PPrivate;
    VariableBuildInfo vbi(m_mesh,m_variable_name,property);
    m_variable_items_local_id = new VariableArrayInt32(vbi);
    m_variable_items_local_id->variable()->setAllocationInfo(DataAllocationInfo(eMemoryLocationHint::HostAndDeviceMostlyRead));
    m_items_local_id = &m_variable_items_local_id->_internalTrueData()->_internalDeprecatedValue();
    updateTimestamp();
  }

  // Regarde si on utilise la version 2 pour ApplyOperationByBasicType
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_APPLYOPERATION_VERSION", true))
    m_use_v2_for_apply_operation = (v.value()==2);

  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DEBUG_APPLYOPERATION", true))
    m_is_debug_apply_operation = (v.value()>0);

  m_is_check_simd_padding = arcaneIsCheck();
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_CHECK_SIMDPADDING", true)){
    m_is_check_simd_padding = (v.value()>0);
    m_is_print_check_simd_padding = (v.value()>1);
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
  m_children_by_type.clear();
  m_children_by_type_ids.clear();
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
checkIsContigous()
{
  m_is_contigous = false;
  Int32ConstArrayView lids = itemsLocalId();
  if (lids.empty()){
    m_is_contigous = false;
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
    m_is_contigous = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroupInternal::
applySimdPadding()
{
  // Fait un padding des derniers éléments du tableau en recopiant la
  // dernière valeurs.
  m_simd_timestamp = timestamp();
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
