// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizer.h                                     (C) 2000-2023 */
/*                                                                           */
/* Interface d'un service de synchronisation des variables.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLESYNCHRONIZER_H
#define ARCANE_CORE_IVARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service de synchronisation de variable.
 *
 * Cette classe est gérée par Arcane et il n'est en générale par nécessaire
 * de l'utiliser directement. Si on souhaite syncrhoniser une variable,
 * il suffit d'appeler la méthode VariableRef::synchronize().
 *
 * Une instance de cette classe est créée via
 * IParallelMng::createVariableSynchronizer(). Une instance est associée
 * à un groupe d'entité. Il faut appeller la fonction compute()
 * pour calculer les infos de synchronisation. Si les entités sont
 * compactées, il faut appeler changeLocalIds().
 */
class ARCANE_CORE_EXPORT IVariableSynchronizer
{
 public:

  virtual ~IVariableSynchronizer() {}

 public:

  //! Gestionnaire parallèle associé
  virtual IParallelMng* parallelMng() = 0;

  /*!
   * \brief Groupe d'entité servant à la synchronisation.
   *
   * L'implémentation actuelle supporte uniquement le groupe
   * de toutes les entités d'une famille.
   */
  virtual const ItemGroup& itemGroup() = 0;

  /*!
   * \brief Recalcule les infos de synchronisation.
   *
   * Cette opération est collective.
   *
   * Cette fonction doit être rappelée si les entités de itemGroup() changent
   * de propriétaire ou si le groupe lui-même évolue.
   * TODO: appeler cette fonction automatiquement si besoin.
   */
  virtual void compute() = 0;

  //! Appelé lorsque les numéros locaux des entités sont modifiés.
  virtual void changeLocalIds(Int32ConstArrayView old_to_new_ids) = 0;

  //! Synchronise la variable \a var en mode bloquant
  virtual void synchronize(IVariable* var) = 0;

  /*!
   * \brief Synchronise les variables \a vars en mode bloquant.
   *
   * Toutes les variables doivent être issues de la même famille
   * et de ce groupe d'entité.
   */
  virtual void synchronize(VariableCollection vars) = 0;

  /*!
   * \brief Rangs des sous-domaines avec lesquels on communique.
   */
  virtual Int32ConstArrayView communicatingRanks() = 0;

  /*!
   * \brief Liste des ids locaux des entités partagées avec un sous-domaine.
   *
   * Le rang du sous-domaine est celui de communicatingRanks()[index].
   */
  virtual Int32ConstArrayView sharedItems(Int32 index) = 0;

  /*!
   * \brief Liste des ids locaux des entités fantômes avec un sous-domaine.
   *
   * Le rang du sous-domaine est celui de communicatingRanks()[index].
   */
  virtual Int32ConstArrayView ghostItems(Int32 index) = 0;

  /*!
   * \brief Synchronise la donnée \a data.
   *
   * La donnée \a data doit être associée à une variable pour laquelle
   * il est valide d'appeler \a synchronize(). Cette méthode est interne
   * à Arcane.
   */
  virtual void synchronizeData(IData* data) = 0;

  /*!
   * \brief Evènement envoyé en début et fin de synchronisation.
   *
   * Cet évènement est envoyé lors des appels aux méthodes
   * de synchronisation synchronize(IVariable* var)
   * et synchronize(VariableCollection vars). Si on souhaite être notifié
   * des synchronisations pour toutes les instances de IVariableSynchronizer,
   * il faut utiliser IVariableMng::synchronizerMng().
   */
  virtual EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
