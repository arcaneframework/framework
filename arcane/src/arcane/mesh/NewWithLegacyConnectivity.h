// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConnectivityNewWithDependenciesInfo.h                       (C) 2000-2024 */
/*                                                                           */
/* Info for new connectivity mode (with dependencies)                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_CONNECTIVITYNEWWITHDEPENDENCIESINFO_H
#define ARCANE_MESH_CONNECTIVITYNEWWITHDEPENDENCIESINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/utils/ArcaneGlobal.h"

#include "arcane/core/IIncrementalItemConnectivity.h"

#include "arcane/mesh/MeshGlobal.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/CompactIncrementalItemConnectivity.h"
#include "arcane/mesh/ItemConnectivitySelector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// todo rename the file in NewWithLegacyConnectivity

/*---------------------------------------------------------------------------*/
/*! \brief class holding a new connectivity but filling also the legacy one
 *  Both custom and legacy connectivities of ItemConnectivitySelector are build.
 *
 */

template <class SourceFamily, class TargetFamily, class LegacyType, class CustomType = IncrementalItemConnectivity>
class ARCANE_MESH_EXPORT NewWithLegacyConnectivity
: public ItemConnectivitySelectorT<LegacyType,CustomType>
, public ReferenceCounterImpl
, public IIncrementalItemConnectivity
{
  ARCCORE_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
public:
  friend class ConnectivityItemVector;

public:
  NewWithLegacyConnectivity(ItemFamily* source_family, IItemFamily* target_family, const String& name)
  : ItemConnectivitySelectorT<LegacyType,CustomType>(source_family,target_family,name)
  {
    //build selector
    Base::template build<SourceFamily,TargetFamily>(); // Create adapted custom connectivity
  }

  typedef ItemConnectivitySelectorT<LegacyType,CustomType> Base;

  String name() const override {return Base::trueCustomConnectivity()->name();}

  bool isEmpty() const {
    return false ;
  }

  //! Liste des familles (sourceFamily() + targetFamily())
  ConstArrayView<IItemFamily*> families() const override {return Base::trueCustomConnectivity()->families();}

  //! Famille source
  IItemFamily* sourceFamily() const override {return Base::trueCustomConnectivity()->sourceFamily();}

  //! Famille cible
  IItemFamily* targetFamily() const override {return Base::trueCustomConnectivity()->targetFamily();}

  //! Ajoute l'entité de localId() \a target_local_id à la connectivité de \a source_item
  void addConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) override {Base::addConnectedItem(source_item,target_local_id);}

  //! Supprime l'entité de localId() \a target_local_id à la connectivité de \a source_item
  void removeConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) override {Base::removeConnectedItem(source_item,target_local_id);}

  //! Supprime toute les entités connectées à \a source_item
  void removeConnectedItems(ItemLocalId source_item) override {Base::removeConnectedItems(source_item);}

  //! Remplace l'entité d'index \a index de \a source_item par l'entité de localId() \a target_local_id
  // Pourquoi cette différence de nom dans ItemConnectivitySelector ?
  void replaceConnectedItem(ItemLocalId source_item,Integer index,ItemLocalId target_local_id) override {Base::replaceItem(source_item,index,target_local_id);}

  //! Remplace les entités de \a source_item par les entités de localId() \a target_local_ids
  // Pourquoi cette différence de nom dans ItemConnectivitySelector ?
  void replaceConnectedItems(ItemLocalId source_item,Int32ConstArrayView target_local_ids) override {Base::replaceItems(source_item,target_local_ids);}

  //! Test l'existence d'un connectivité entre \a source_item et l'entité de localId() \a target_local_id
  bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const override { return Base::hasConnectedItem(source_item,target_local_id);};

  //TODO: utiliser un mécanisme par évènement.
  //! Notifie la connectivité que la famille source est compactée.
  void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) override {Base::trueCustomConnectivity()->notifySourceFamilyLocalIdChanged(new_to_old_ids);}

  //TODO: utiliser un mécanisme par évènement.
  //! Notifie la connectivité que la famille cible est compactée.
  void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) override {Base::trueCustomConnectivity()->notifyTargetFamilyLocalIdChanged(old_to_new_ids);}

  //! Notifie la connectivité qu'une entité a été ajoutée à la famille source.
  void notifySourceItemAdded(ItemLocalId item) override {Base::trueCustomConnectivity()->notifySourceItemAdded(item);}

  //! Notifie la connectivité qu'on a effectué une relecture à partir d'une protection
  void notifyReadFromDump() override {Base::trueCustomConnectivity()->notifyReadFromDump();}

  //! Nombre d'entités préalloués pour la connectivité de chaque entité
  Integer preAllocatedSize() const override {return Base::preAllocatedSize();}

  //! Positionne le nombre d'entités à préallouer pour la connectivité de chaque entité
  void setPreAllocatedSize(Integer value) override {Base::setPreAllocatedSize(value);}

  //! Sort sur le flot \a out des statistiques sur l'utilisation et la mémoire utilisée
  void dumpStats(std::ostream& out) const override { Base::trueCustomConnectivity()->dumpStats(out); }

  //! Nombre d'entité connectées à l'entité source de numéro local \a lid
  Integer nbConnectedItem(ItemLocalId lid) const override {return Base::trueCustomConnectivity()->nbConnectedItem(lid);}

  //! localId() de la \a index-ième entitée connectées à l'entité source de numéro local \a lid
  Int32 connectedItemLocalId(ItemLocalId lid,Integer index) const override  {return Base::trueCustomConnectivity()->connectedItemLocalId(lid,index);}

  //! Nombre maximum d'entités connectées à une entité source.
  Int32 maxNbConnectedItem() const override { return Base::trueCustomConnectivity()->maxNbConnectedItem(); }

  Ref<IIncrementalItemSourceConnectivity> toSourceReference() override
  {
    return Arccore::makeRef<IIncrementalItemSourceConnectivity>(this);
  }
  Ref<IIncrementalItemTargetConnectivity> toTargetReference() override
  {
    return Arccore::makeRef<IIncrementalItemTargetConnectivity>(this);
  }
  IIncrementalItemConnectivityInternal* _internalApi() override
  {
    return Base::trueCustomConnectivity()->_internalApi();
  }

 protected:

  //! Implémente l'initialisation de \a civ pour cette connectivitée.
  void _initializeStorage(ConnectivityItemVector* civ) override {Base::trueCustomConnectivity()->_initializeStorage(civ);};

  //! Remplit \a con_items avec les entités connectées à \a item.
  ItemVectorView _connectedItems(ItemLocalId item,ConnectivityItemVector& con_items) const override {return Base::trueCustomConnectivity()->_connectedItems(item,con_items);}

};


template <class SourceFamily, class TargetFamily>
class ARCANE_MESH_EXPORT NewWithLegacyConnectivityType
{
public:
  typedef NewWithLegacyConnectivity<SourceFamily,TargetFamily,typename LegacyConnectivityTraitsT<TargetFamily>::type> type;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* CONNECTIVITYNEWWITHDEPENDENCIESINFO_H_ */
