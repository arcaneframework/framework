// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupInternal.h                                         (C) 2000-2025 */
/*                                                                           */
/* Partie interne à Arcane de ItemGroup.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ITEMGROUPINTERNAL_H
#define ARCANE_CORE_INTERNAL_ITEMGROUPINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/internal/ItemGroupImplInternal.h"

#include <map>
#include <mutex>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemGroupImpl;

/*!
 * \brief Gestion de partition d'un groupe suivant le type de ses éléments.
 */
class ItemGroupChildrenByType
{
 public:

  explicit ItemGroupChildrenByType(ItemGroupInternal* igi)
  : m_group_internal(igi)
  {}

 public:

  void clear()
  {
    m_children_by_type.clear();
    m_children_by_type_ids.clear();
  }
  void applyOperation(IItemOperationByBasicType* operation);
  bool isUseV2ForApplyOperation() const { return m_use_v2_for_apply_operation; }
  void _initChildrenByTypeV2();
  void _computeChildrenByTypeV2();
  void _initChildrenByType();
  void _computeChildrenByType();

 public:

  //! Vrai si on utilise la version 2 de la gestion pour applyOperation().
  bool m_use_v2_for_apply_operation = true;

 public:

  /*!
   * \brief Liste des localId() par type d'entité.
   *
   * Ce champ est utilisé avec la version 2.
   */
  UniqueArray<UniqueArray<Int32>> m_children_by_type_ids;

  /*!
   * \brief Liste des fils de ce groupe par type d'entité.
   *
   * Ce champ est utilisé avec la version 1 qui demande
   * de créer un groupe par sous-type.
   */
  UniqueArray<ItemGroupImpl*> m_children_by_type;

  /*!
   * \brief Indique le type des entités du groupe.
   *
   * Si différent de IT_NullType, cela signifie que toutes
   * les entités du groupe sont du même type et donc il n'est
   * pas nécessaire de calculer le localId() des entités par type.
   * On utilise dans ce cas directement le groupe en paramètre
   * des applyOperation().
   */
  ItemTypeId m_unique_children_type{ IT_NullType };

  //! Timestamp indiquant quand a été calculé la liste des ids des enfants
  Int64 m_children_by_type_ids_computed_timestamp = -1;

  bool m_is_debug_apply_operation = false;

  ItemGroupInternal* m_group_internal = nullptr;
  ItemGroupImpl* m_group_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation de la classe ItemGroupImpl.
 *
 * Le container contenant la liste des entités du groupe est soit une
 * variable dans le cas d'une group standard, soit un tableau simple
 * dans le cas d'un groupe ayant un parent. En effet, les groupes
 * ayant des parents sont des groupes générés dynamiquement (par
 * exemple le groupe des entités propres) et ne sont donc pas
 * toujours présents sur tous les sous-domaines (une variable doit toujours
 * exister sur tous les sous-domaines). De plus, leur valeur n'a
 * pas besoin d'être sauvée lors d'une protection.

 \todo ajouter notion de groupe généré, avec les propriétés suivantes:
 - ces groupes ne doivent pas être transférés d'un sous-domaine à l'autre
 - ils ne peuvent pas être modifiés directement.
 */
class ItemGroupInternal
{
  friend class ItemGroupImplInternal;

 public:

  /*!
   * \brief Mutex pour protéger les appels à ItemGroupImpl::_checkNeedUpdate().
   *
   * Par défaut le mutex n'est pas actif. Il faut appeler create() pour le
   * rendre actif.
   */
  class CheckNeedUpdateMutex
  {
   public:

    class ScopedLock
    {
     public:

      explicit ScopedLock(const CheckNeedUpdateMutex& mutex)
      : m_update_mutex(mutex)
      {
        m_update_mutex._lock();
      }
      ~ScopedLock()
      {
        m_update_mutex._unlock();
      }

     private:

      const CheckNeedUpdateMutex& m_update_mutex;
    };

   public:

    ~CheckNeedUpdateMutex()
    {
      delete m_mutex;
    }
    void create()
    {
      m_mutex = new std::mutex();
    }

   private:

    std::mutex* m_mutex = nullptr;

   private:

    void _lock() const
    {
      if (m_mutex)
        m_mutex->lock();
    }
    void _unlock() const
    {
      if (m_mutex)
        m_mutex->unlock();
    }
  };

 public:

  ItemGroupInternal();
  ItemGroupInternal(IItemFamily* family, const String& name);
  ItemGroupInternal(IItemFamily* family, ItemGroupImpl* parent, const String& name);
  ~ItemGroupInternal();

 public:

  const String& name() const { return m_name; }
  const String& fullName() const { return m_full_name; }
  bool null() const { return m_is_null; }
  IMesh* mesh() const { return m_mesh; }
  eItemKind kind() const { return m_kind; }
  Integer maxLocalId() const;
  ItemInternalList items() const;
  ItemInfoListView itemInfoListView() const;

  Int32ArrayView itemsLocalId() { return *m_items_local_id; }
  Int32ConstArrayView itemsLocalId() const { return *m_items_local_id; }
  Int32Array& mutableItemsLocalId() { return *m_items_local_id; }
  VariableArrayInt32* variableItemsLocalid() { return m_variable_items_local_id; }

  Int64 timestamp() const { return m_timestamp; }
  bool isContiguous() const { return m_is_contiguous; }
  void checkIsContiguous();

  void updateTimestamp()
  {
    ++m_timestamp;
    m_is_contiguous = false;
  }

  void setNeedRecompute()
  {
    // NOTE: normalement, il ne faudrait mettre cette valeur à 'true' que pour
    // les groupes recalculés (qui ont un parent ou pour lequel 'm_compute_functor' n'est
    // pas nul). Cependant, cette méthode est aussi appelé sur le groupe de toutes les entités
    // et peut-être d'autres groupes.
    // Changer ce comportement risque d'impacter pas mal de code donc il faudrait bien vérifier
    // que tout est OK avant de faire cette modification.
    m_need_recompute = true;
  }

  //! Applique le padding pour la vectorisation
  void applySimdPadding();

  void checkUpdateSimdPadding();
  bool isAllItems() const { return m_is_all_items; }
  bool isOwn() const { return m_is_own; }
  Int32 nbItem() const { return itemsLocalId().size(); }
  void checkValid();

 public:

  void _removeItems(SmallSpan<const Int32> items_local_id);

 private:

  void _notifyDirectRemoveItems(SmallSpan<const Int32> removed_ids, Int32 nb_remaining);

 public:

  ItemGroupImplInternal m_internal_api;
  IMesh* m_mesh = nullptr; //!< Gestionnaire de groupe associé
  IItemFamily* m_item_family = nullptr; //!< Famille associée
  ItemGroupImpl* m_parent = nullptr; //! Groupe parent (groupe null si aucun)
  String m_variable_name; //!< Nom de la variable contenant les indices des éléments du groupe
  String m_full_name; //!< Nom complet du groupe.
  bool m_is_null = true; //!< \a true si le groupe est nul
  eItemKind m_kind = IK_Unknown; //!< Genre des entités du groupe
  String m_name; //!< Nom du groupe
  bool m_is_own = false; //!< \a true si groupe contient uniquement les entités dont on est propriétaire.

 private:

  Int64 m_timestamp = -1; //!< Temps de la derniere modification

 public:

  Int64 m_simd_timestamp = -1; //!< Temps de la derniere modification pour le calcul des infos SIMD
  //@{ @name alias locaux à des sous-groupes de m_sub_groups
  ItemGroupImpl* m_own_group = nullptr; //!< Items owned by the subdomain
  ItemGroupImpl* m_ghost_group = nullptr; //!< Items not owned by the subdomain
  ItemGroupImpl* m_interface_group = nullptr; //!< Items on the boundary of two subdomains
  ItemGroupImpl* m_node_group = nullptr; //!< Groupe des noeuds
  ItemGroupImpl* m_edge_group = nullptr; //!< Groupe des arêtes
  ItemGroupImpl* m_face_group = nullptr; //!< Groupe des faces
  ItemGroupImpl* m_cell_group = nullptr; //!< Groupe des mailles
  ItemGroupImpl* m_inner_face_group = nullptr; //!< Groupe des faces internes
  ItemGroupImpl* m_outer_face_group = nullptr; //!< Groupe des faces externes
  //! AMR
  // FIXME on peut éviter de stocker ces groupes en introduisant des predicats
  // sur les groupes parents
  ItemGroupImpl* m_active_cell_group = nullptr; //!< Groupe des mailles actives
  ItemGroupImpl* m_own_active_cell_group = nullptr; //!< Groupe des mailles propres actives
  ItemGroupImpl* m_active_face_group = nullptr; //!< Groupe des faces actives
  ItemGroupImpl* m_own_active_face_group = nullptr; //!< Groupe des faces actives propres
  ItemGroupImpl* m_inner_active_face_group = nullptr; //!< Groupe des faces internes actives
  ItemGroupImpl* m_outer_active_face_group = nullptr; //!< Groupe des faces externes actives
  std::map<Integer, ItemGroupImpl*> m_level_cell_group; //!< Groupe des mailles de niveau
  std::map<Integer, ItemGroupImpl*> m_own_level_cell_group; //!< Groupe des mailles propres de niveau

  //@}
  std::map<String, AutoRefT<ItemGroupImpl>> m_sub_groups; //!< Ensemble de tous les sous-groupes
  bool m_need_recompute = false; //!< Vrai si le groupe doit être recalculé
  bool m_need_invalidate_on_recompute = false; //!< Vrai si l'on doit activer les invalidate observers en cas de recalcul
  bool m_transaction_mode = false; //!< Vrai si le groupe est en mode de transaction directe
  bool m_is_local_to_sub_domain = false; //!< Vrai si le groupe est local au sous-domaine
  IFunctor* m_compute_functor = nullptr; //!< Fonction de calcul du groupe
  bool m_is_all_items = false; //!< Indique s'il s'agit du groupe de toutes les entités
  bool m_is_constituent_group = false; //!< Indique si le groupe est associé à un constituant (IMeshComponent)
  SharedPtrT<GroupIndexTable> m_group_index_table; //!< Table de hachage du local id des items vers leur position en enumeration
  Ref<IVariableSynchronizer> m_synchronizer; //!< Synchronizer du groupe

  // Anciennement dans DynamicMeshKindInfo
  UniqueArray<Int32> m_items_index_in_all_group; //! localids -> index (UNIQUEMENT ALLITEMS)

  std::map<const void*, IItemGroupObserver*> m_observers; //!< Observers du groupe
  bool m_observer_need_info = false; //!< Synthése de besoin de observers en informations de transition
  void notifyExtendObservers(const Int32ConstArrayView* info);
  void notifyReduceObservers(const Int32ConstArrayView* info);
  void notifyCompactObservers(const Int32ConstArrayView* info);
  void notifyInvalidateObservers();

  void resetSubGroups();

 public:

  UniqueArray<Int32> m_local_buffer{ MemoryUtils::getAllocatorForMostlyReadOnlyData() };
  Array<Int32>* m_items_local_id = &m_local_buffer; //!< Liste des numéros locaux des entités de ce groupe
  VariableArrayInt32* m_variable_items_local_id = nullptr;
  bool m_is_contiguous = false; //! Vrai si les localIds sont consécutifs.
  bool m_is_check_simd_padding = true;
  bool m_is_print_check_simd_padding = false;
  bool m_is_print_apply_simd_padding = false;
  bool m_is_print_stack_apply_simd_padding = false;

 public:

  //! Mutex pour protéger la mise à jour.
  CheckNeedUpdateMutex m_check_need_update_mutex;

 public:

  //! Sous-partie d'un groupe en fonction de son type
  ItemGroupChildrenByType m_sub_parts_by_type;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
