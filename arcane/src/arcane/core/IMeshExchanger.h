// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshExchanger.h                                            (C) 2000-2022 */
/*                                                                           */
/* Gestion d'un échange de maillage entre sous-domaines.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHEXCHANGER_H
#define ARCANE_IMESHEXCHANGER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IPrimaryMesh;
class IItemFamily;
class IItemFamilyExchanger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'un échange de maillage entre sous-domaines.
 *
 * Un échange se fait en plusieurs phases, qui doivent être effectuées
 * dans l'ordre dicté par l'énumération ePhase.
 *
 */
class ARCANE_CORE_EXPORT IMeshExchanger
{
 public:
  //! Indique les différentes phases de l'échange
  enum class ePhase
  {
    Init = 0,
    ComputeInfos,
    ProcessExchange,
    RemoveItems,
    AllocateItems,
    UpdateItemGroups,
    UpdateVariables,
    Finalize,
    Ended
  };
 public:

  virtual ~IMeshExchanger() {} //<! Libère les ressources

 public:

  /*!
   * \brief Calcule les infos à envoyer/recevoir des autres sous-domaines.
   *
   * Cette opération est collective.
   *
   * Le calcul des informations à envoyer se fait en connaissant le nouveau
   * propriétaire de chaque entité. Cette information est conservée dans
   * la variable IItemFamily::itemsNewOwner(). Par exemple, une maille
   * sera migrée si le nouveau propriétaire est différent du propriétaire
   * actuel (qui est donné par Item::owner()).
   *
   * Après appel à cette méthode chaque entité du maillage est modifiée comme suit:
   * - le champ Item::owner() indique le nouveau propriétaire.
   * - les entités qui seront supprimées après l'échange sont marquées par le flag
   *   ItemFlags::II_NeedRemove (sauf pour l'instant pour les particules
   *   sans notion de fantôme mais c'est temporaire).
   *
   * Retourne \a true s'il n'y a aucun échange à effectuer.
   *
   * \pre phase()==ePhase::ComputeInfos
   * \post phase()==ePhase::ProcessExchange
   */
  virtual bool computeExchangeInfos() =0;

  /*!
   * \brief Procède à l'échange des informations entre les sous-domaines.
   *
   * Cette opération est collective.
   *
   * Cette opération ne fait aucune modification sur le maillage. Elle se
   * contente juste d'envoyer et de recevoir les informations nécesaire pour
   * la mise à jour du maillage.
   *
   * \pre phase()==ePhase::ProcessExchange
   * \post phase()==ePhase::RemoveItems
   */
  virtual void processExchange() =0;

  /*!
   * \brief Supprime de ce sous-domaine les entités qui ne doivent plus
   * s'y trouver suite à l'échange.
   *
   * Toutes les entités marquées avec le flag ItemFlags::II_NeedRemove
   * sont supprimées.
   *
   * \pre phase()==ePhase::RemoveItems
   * \post phase()==ePhase::AllocateItems
   */
  virtual void removeNeededItems() =0;

  /*!
   * \brief Alloue les entités réceptionnées depuis les autre sous-domaines.
   *
   * Cette opération est collective.
   *
   * \pre phase()==ePhase::AllocateItems
   * \post phase()==ePhase::UpdateItemGroups
   */
  virtual void allocateReceivedItems() =0;

  /*!
   * \brief Mise à jour des groupes d'entités
   *
   * Cette opération est collective.
   *
   * \pre phase()==ePhase::UpdateItemGroups
   * \post phase()==ePhase::UpdateVariables
   */
  virtual void updateItemGroups() =0;

  /*!
   * \brief Mise à jour des variables
   *
   * Cette opération est collective.
   *
   * \pre phase()==ePhase::UpdateVariables
   * \post phase()==ePhase::Finalize
   */
  virtual void updateVariables() =0;

  /*!
   * \brief Finalise les échanges.
   *
   * Cette opération est collective.
   *
   * Cette méthode effectue les dernières opérations nécessaires lors
   * de l'échange.
   *
   * \pre phase()==ePhase::Finalize
   * \post phase()==ePhase::Ended
   */
  virtual void finalizeExchange() =0;

  //! Maillage associé à cet échangeur.
  virtual IPrimaryMesh* mesh() const =0;

  //! Échangeur associé à la famille \a family. Lance une exception si non trouvé
  virtual IItemFamilyExchanger* findExchanger(IItemFamily* family) =0;

  //! Phase de l'échange dans laquelle on se trouve.
  virtual ePhase phase() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
