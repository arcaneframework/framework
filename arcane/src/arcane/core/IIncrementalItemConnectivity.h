// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIncrementalItemConnectivity.h                              (C) 2000-2024 */
/*                                                                           */
/* Interface de connectivité incrémentale des entités.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IINCREMENTALITEMCONNECTIVITY_H
#define ARCANE_CORE_IINCREMENTALITEMCONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface de la source d'une connectivité incrémentale
 */
class ARCANE_CORE_EXPORT IIncrementalItemSourceConnectivity
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IIncrementalItemSourceConnectivity() = default;

 public:

  //! Famille source
  virtual IItemFamily* sourceFamily() const = 0;

  //TODO: utiliser un mécanisme par évènement.
  //! Notifie la connectivité que la famille source est compactée.
  virtual void notifySourceFamilyLocalIdChanged(Int32ConstArrayView new_to_old_ids) = 0;

  //! Notifie la connectivité qu'une entité a été ajoutée à la famille source.
  virtual void notifySourceItemAdded(ItemLocalId item) = 0;

  /*!
   * \brief Réserve la mémoire pour \a n entités sources.
   *
   * L'appel à cette méthode est optionnel mais permet d'éviter de multiples
   * réallocations lors d'appels successifs à notifySourceItemAdded().
   *
   * Si \a pre_alloc_connectivity est vrai, pré-alloue aussi les la liste des
   * connectivités en fonction de la valeur de preAllocatedSize(). Par exemple
   * si preAllocatedSize() vaut 4 et si \a n vaut 10000, on va pré-allouer
   * pour 40000 connectivités. Pour éviter une surconsommation mémoire inutile,
   * il ne faut pré-allouer les connectivités que si on est sur qu'on va les utiliser.
   */
  virtual void reserveMemoryForNbSourceItems(Int32 n, bool pre_alloc_connectivity);

  //! Notifie la connectivité qu'on a effectué une relecture à partir d'une protection
  virtual void notifyReadFromDump() = 0;

  //! Retourne une référence sur l'instance
  virtual Ref<IIncrementalItemSourceConnectivity> toSourceReference() = 0;

 private:

  // Interfaces réservées à Arcane

  //! Notifie la connectivité que les entités \a items ont été ajoutées à la famille source
  virtual void _internalNotifySourceItemsAdded(Int32ConstArrayView items);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface de la cible d'une connectivité incrémentale
 */
class ARCANE_CORE_EXPORT IIncrementalItemTargetConnectivity
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 protected:

  virtual ~IIncrementalItemTargetConnectivity() = default;

 public:

  //TODO: utiliser un mécanisme par évènement.
  //! Notifie la connectivité que la famille cible est compactée.
  virtual void notifyTargetFamilyLocalIdChanged(Int32ConstArrayView old_to_new_ids) = 0;

  //! Retourne une référence sur l'instance
  virtual Ref<IIncrementalItemTargetConnectivity> toTargetReference() = 0;
};

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
, public IIncrementalItemSourceConnectivity
, public IIncrementalItemTargetConnectivity
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  //TODO rendre 'protected' une fois que tout le monde utilisera le compteur de référence
  ~IIncrementalItemConnectivity() = default;

 public:

  //! Nom de la connectivité
  virtual String name() const = 0;

  //! Liste des familles (sourceFamily() + targetFamily())
  virtual ConstArrayView<IItemFamily*> families() const = 0;

  //! Famille cible
  virtual IItemFamily* targetFamily() const = 0;

  //! Ajoute l'entité de localId() \a target_local_id à la connectivité de \a source_item
  virtual void addConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) = 0;

  /*!
   * \brief Alloue et positionne les entités connectées à \a source_item.
   *
   * S'il y avait des déjà des entités connectées à \a source_item, elles sont supprimées.
   * \a target_local_ids contient la liste des numéros locaux des entités à ajouter.
   * Cette méthode est équivalente à appeler le code suivant mais permet des optimisations sur la
   * gestion mémoire:
   * \code
   * IIncrementalItemConnectivity* c = ...;
   * c->removeConnectedItems(source_item);
   * for( Int32 x : target_local_ids )
   *   c->addConnectedItem(source_item,ItemLocalId{x});
   * \endcode
   */
  virtual void setConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids);

  //! Supprime l'entité de localId() \a target_local_id à la connectivité de \a source_item
  virtual void removeConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) = 0;

  //! Supprime toute les entités connectées à \a source_item
  virtual void removeConnectedItems(ItemLocalId source_item) = 0;

  //! Remplace l'entité d'index \a index de \a source_item par l'entité de localId() \a target_local_id
  virtual void replaceConnectedItem(ItemLocalId source_item, Integer index, ItemLocalId target_local_id) = 0;

  //! Remplace les entités de \a source_item par les entités de localId() \a target_local_ids
  virtual void replaceConnectedItems(ItemLocalId source_item, Int32ConstArrayView target_local_ids) = 0;

  //! Test l'existence d'un connectivité entre \a source_item et l'entité de localId() \a target_local_id
  virtual bool hasConnectedItem(ItemLocalId source_item, ItemLocalId target_local_id) const = 0;

  /*!
   * \brief Nombre maximum d'entités connectées à une entité source.
   *
   * Cette valeur peut être supérieure au nombre maximum actuel d'entités
   * connectées s'il y a eu des appels à removeConnectedItem() et removeConnectedItems().
   */
  virtual Int32 maxNbConnectedItem() const = 0;

  //! Nombre d'entités pré-alloués pour la connectivité de chaque entité
  virtual Integer preAllocatedSize() const = 0;

  //! Positionne le nombre d'entités à pré-allouer pour la connectivité de chaque entité
  virtual void setPreAllocatedSize(Integer value) = 0;

  //! Sort sur le flot \a out des statistiques sur l'utilisation et la mémoire utilisée
  virtual void dumpStats(std::ostream& out) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
