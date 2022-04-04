﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIncrementalItemConnectivity.h                              (C) 2000-2018 */
/*                                                                           */
/* Interface de connectivité incrémentale des entités.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IINCREMENTALITEMCONNECTIVITY_H
#define ARCANE_IINCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/ItemTypes.h"
#include "arcane/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer une connectivité incrémentale.
 *
 * Une connectivité relie deux familles, une source (sourceFamily()) et
 * une cible (targetFamily()).
 */
class ARCANE_CORE_EXPORT IIncrementalItemConnectivity
: public IItemConnectivityAccessor
{
 public:

  virtual ~IIncrementalItemConnectivity(){}

 public:

  //! Nom de la connectivité
  virtual const String& name() const =0;

  //! Liste des familles (sourceFamily() + targetFamily())
  virtual ConstArrayView<IItemFamily*> families() const =0;

  //! Famille source
  virtual IItemFamily* sourceFamily() const =0;

  //! Famille cible
  virtual IItemFamily* targetFamily() const =0;

  //! Ajoute l'entité de localId() \a target_local_id à la connectivité de \a source_item
  virtual void addConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) =0;

  //! Supprime l'entité de localId() \a target_local_id à la connectivité de \a source_item
  virtual void removeConnectedItem(ItemLocalId source_item,ItemLocalId target_local_id) =0;

  //! Supprime toute les entités connectées à \a source_item
  virtual void removeConnectedItems(ItemLocalId source_item) =0;

  //! Remplace l'entité d'index \a index de \a source_item par l'entité de localId() \a target_local_id
  virtual void replaceConnectedItem(ItemLocalId source_item,Integer index,ItemLocalId target_local_id) =0;

  //! Remplace les entités de \a source_item par les entités de localId() \a target_local_ids
  virtual void replaceConnectedItems(ItemLocalId source_item,Int32ConstArrayView target_local_ids) =0;

  //! Test l'existence d'un connectivité entre \a source_item et l'entité de localId() \a target_local_id
  virtual bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const =0;

  //TODO: utiliser un mécanisme par évènement.
  //! Notifie la connectivité que la famille source est compactée.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) =0;

  //TODO: utiliser un mécanisme par évènement.
  //! Notifie la connectivité que la famille cible est compactée.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) =0;

  //! Notifie la connectivité qu'une entité a été ajoutée à la famille source.
  virtual void notifySourceItemAdded(ItemLocalId item) =0;

  //! Notifie la connectivité qu'on a effectué une relecture à partir d'une protection
  virtual void notifyReadFromDump() =0;

  //! Nombre d'entités préalloués pour la connectivité de chaque entité
  virtual Integer preAllocatedSize() const =0;

  //! Positionne le nombre d'entités à préallouer pour la connectivité de chaque entité
  virtual void setPreAllocatedSize(Integer value) =0;

  //! Sort sur le flot \a out des statistiques sur l'utilisation et la mémoire utilisée
  virtual void dumpStats(std::ostream& out) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
