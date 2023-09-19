// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableSynchronizerMng.h                                  (C) 2000-2023 */
/*                                                                           */
/* Interface du gestionnaire de synchronisation des variables.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLESYNCHRONIZERMNG_H
#define ARCANE_CORE_IVARIABLESYNCHRONIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariableSynchronizerMngInternal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de synchronisation des variables.
 */
class ARCANE_CORE_EXPORT IVariableSynchronizerMng
{
 public:

  virtual ~IVariableSynchronizerMng() = default;

 public:

  //! Gestionnaire de parallélisme associé
  virtual IParallelMng* parallelMng() const = 0;

  /*!
   * \brief Evènement envoyé en début et fin de synchronisation.
   *
   * Cet évènement est envoyé lors des appels aux méthodes
   * de synchronisation IVariableSynchronizer::synchronize(IVariable* var)
   * et IVariableSynchronizer::synchronize(VariableCollection vars) pour toutes
   * les instances de IVariableSynchronizer.
   */
  virtual EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() = 0;

  /*!
   * \brief Indique si on compare les valeurs avant et après synchronisations.
   *
   * L'état d'activation doit être le même sur l'ensemble des rangs de parallelMng().
   */
  virtual void setCompareSynchronize(bool v) = 0;

  //! Indique si on effecture les comparaisons des valeurs avant et après synchronisation
  virtual bool isCompareSynchronize() const = 0;

  /*!
   * \brief Affiche les statistiques sur le flot \a ostr.
   *
   * Il faut avoir traiter les statistiques via l'appel à flushPendingStats()
   * avant d'appeler cette méthode
   */
  virtual void dumpStats(std::ostream& ostr) const = 0;

  /*!
   * \brief Traite les statistiques en cours.
   *
   * Cette méthode ne fait rien si isCompareSynchronize() vaut \a false.
   *
   * Cette méthode est collective sur parallelMng().
   */
  virtual void flushPendingStats() = 0;

 public:

  virtual IVariableSynchronizerMngInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
