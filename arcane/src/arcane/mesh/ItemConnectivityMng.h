// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityMng.h                                       (C) 2000-2023 */
/*                                                                           */
/* Gestionnaire des connectivités                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CONNECTIVITYMANAGER_H
#define ARCANE_CONNECTIVITYMANAGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Array.h"

#include "arcane/IItemConnectivity.h"
#include "arcane/IItemConnectivityMng.h"
#include "arcane/IItemConnectivitySynchronizer.h"
#include "arcane/IIncrementalItemConnectivity.h"
#include "arcane/IItemFamily.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemConnectivityGhostPolicy;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FamilyState
{
 public:
  FamilyState() : m_state(-1)    {}
  FamilyState(const FamilyState&) = default;
  virtual ~FamilyState() = default;

 public:
  //! Concatenation of all the family changes during the simulation
  Int32SharedArray m_added_items;
  //! idem
  Int32SharedArray m_removed_items;
  //! Current added items (no history)
  Int32UniqueArray m_current_added_items;
  /*! incremented at each change. Used to know if the connectivity is up
    to date with the family */
  Integer m_state;
  /*! Indicate the position in added or removed_items arrays of
    the first item added or removed in the current state. */
  IntegerSharedArray  m_state_first_added_item_index;
  /*! Indicate the position in added or removed_items arrays of the first
    item added or removed in the current state. */
  IntegerSharedArray  m_state_first_removed_item_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ConnectivityStateData
{
public:
  ConnectivityStateData() : m_last_family_state(-1), m_last_added_item_index(-1), m_last_removed_item_index(-1) {}
  ConnectivityStateData(const ConnectivityStateData&) = default;
  virtual ~ConnectivityStateData() = default;

public :
  Integer m_last_family_state;
  Integer m_last_added_item_index;
  Integer m_last_removed_item_index;
};

class ConnectivityState
{
 public:
  ConnectivityState(){}
  ConnectivityState(const ConnectivityState&) = default;
  virtual ~ConnectivityState() = default;

 public :
  ConnectivityStateData m_state_with_source_family;
  ConnectivityStateData m_state_with_target_family;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemConnectivityMng
: public IItemConnectivityMng
{
public:

  /** Constructeur de la classe */
  ItemConnectivityMng(ITraceMng* trace_mng) : m_trace_mng(trace_mng) {}

  /** Destructeur de la classe */
  ~ItemConnectivityMng() override
  {
    for (const auto& map_element : m_synchronizers) {
      if (map_element.second) delete map_element.second;
    }
  }

  // Save Connectivity
  void registerConnectivity(IItemConnectivity* connectivity) override
  {
    connectivity->sourceFamily()->addSourceConnectivity(connectivity);
    connectivity->targetFamily()->addTargetConnectivity(connectivity);
    // refactoring
    connectivity->sourceFamily()->setConnectivityMng(this);
    connectivity->targetFamily()->setConnectivityMng(this);
    _register(connectivity->name(),connectivity->sourceFamily()->fullName(),connectivity->targetFamily()->fullName());
  }

  void unregisterConnectivity(IItemConnectivity* connectivity) override
  {
    connectivity->sourceFamily()->removeSourceConnectivity(connectivity);
    connectivity->targetFamily()->removeTargetConnectivity(connectivity);
  }

  // Save Connectivity
  void registerConnectivity(IIncrementalItemConnectivity* connectivity) override
  {
    //connectivity->sourceFamily()->addSourceConnectivity(connectivity);
    //connectivity->targetFamily()->addTargetConnectivity(connectivity);
    // refactoring
    connectivity->sourceFamily()->setConnectivityMng(this);
    connectivity->targetFamily()->setConnectivityMng(this);
    _register(connectivity->name(),connectivity->sourceFamily()->fullName(),connectivity->targetFamily()->fullName());
  }

  void unregisterConnectivity([[maybe_unused]] IIncrementalItemConnectivity* connectivity) override
  {
    //connectivity->sourceFamily()->removeSourceConnectivity(connectivity);
    //connectivity->targetFamily()->removeTargetConnectivity(connectivity);
  }


  /*! \brief Création d'un objet de synchronisation pour une connectivité.
   *
   *  Si la méthode a déjà été appelée pour cette connectivité, un nouveau synchroniseur est créé et le précedent est détruit.
   *
   */
  IItemConnectivitySynchronizer* createSynchronizer(IItemConnectivity* connectivity,
                                                    IItemConnectivityGhostPolicy* connectivity_ghost_policy) override;
  IItemConnectivitySynchronizer* getSynchronizer(IItemConnectivity* connectivity) override
  {
    // TODO handle failure
    return m_synchronizers[connectivity];
  }

  //! Enregistrement de modifications d'une famille d'items
  void setModifiedItems(IItemFamily* family, Int32ConstArrayView added_items,Int32ConstArrayView removed_items) override;

  //! Mise à jour des items modifiés éventuellement compactés
  void notifyLocalIdChanged(IItemFamily* family, Int32ConstArrayView old_to_new_ids,Integer nb_item) override;

  //! Test si la connectivité est à jour par rapport à la famille source et à la famille target
  bool isUpToDate(IItemConnectivity* connectivity) override
  {
    return (isUpToDateWithSourceFamily(connectivity) && isUpToDateWithTargetFamily(connectivity));
  }
  bool isUpToDateWithSourceFamily(IItemConnectivity* connectivity) override
  {
    return (_lastUpdateSourceFamilyState(connectivity->name()) == _familyState(connectivity->sourceFamily()->fullName()));
  }
  bool isUpToDateWithTargetFamily(IItemConnectivity* connectivity) override
  {
    return (_lastUpdateTargetFamilyState(connectivity->name()) == _familyState(connectivity->targetFamily()->fullName()));
  }

  //! Enregistre la connectivité comme mise à jour par rapport aux deux familles (source et target)
  void setUpToDate(IItemConnectivity* connectivity) override;

  //! Test si la connectivité est à jour par rapport à la famille source et à la famille target
    bool isUpToDate(IIncrementalItemConnectivity* connectivity) override
    {
      return (isUpToDateWithSourceFamily(connectivity) && isUpToDateWithTargetFamily(connectivity));
    }
    bool isUpToDateWithSourceFamily(IIncrementalItemConnectivity* connectivity) override
    {
      return (_lastUpdateSourceFamilyState(connectivity->name()) == _familyState(connectivity->sourceFamily()->fullName()));
    }
    bool isUpToDateWithTargetFamily(IIncrementalItemConnectivity* connectivity) override
    {
      return (_lastUpdateTargetFamilyState(connectivity->name()) == _familyState(connectivity->targetFamily()->fullName()));
    }

    //! Enregistre la connectivité comme mise à jour par rapport aux deux familles (source et target)
    void setUpToDate(IIncrementalItemConnectivity* connectivity) override;

  //! Récupération des items modifiés pour mettre à jour une connectivité
  void getSourceFamilyModifiedItems(IItemConnectivity* connectivity, Int32ArrayView& added_items,
                                    Int32ArrayView& removed_items) override;
  void getTargetFamilyModifiedItems(IItemConnectivity* connectivity, Int32ArrayView& added_items,
                                    Int32ArrayView& removed_items) override;


  void getSourceFamilyModifiedItems(IIncrementalItemConnectivity* connectivity, Int32ArrayView& added_items,
                                    Int32ArrayView& removed_items) override;
  void getTargetFamilyModifiedItems(IIncrementalItemConnectivity* connectivity, Int32ArrayView& added_items,
                                    Int32ArrayView& removed_items) override;

 private:

  void _register(const String& connectivity_name,const String& from_family_name, const String& to_family_name);
  Integer _lastUpdateSourceFamilyState(const String& connectivity_name);
  Integer _lastUpdateTargetFamilyState(const String& connectivity_name);
  Integer _familyState(const String& family_name);
  ConnectivityState& _findConnectivity(const String& connectivity_name);
  FamilyState&       _findFamily(const String& family_full_name);
  void _getModifiedItems(ConnectivityStateData& connectivity_state, FamilyState& family_state, Int32ArrayView& added_items, Int32ArrayView& removed_items);
  void _setUpToDate(ConnectivityStateData& connectivity_state, FamilyState& family_state);

  ITraceMng* m_trace_mng;
  std::map<IItemConnectivity*,IItemConnectivitySynchronizer*> m_synchronizers;
  typedef std::map<const String,FamilyState> FamilyStateMap;
  FamilyStateMap m_family_states;
  typedef std::map<const String,ConnectivityState> ConnectivityStateMap;
  ConnectivityStateMap m_connectivity_states;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* CONNECTIVITYMANAGER_H_ */
