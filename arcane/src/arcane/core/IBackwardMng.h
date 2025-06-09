// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IBackwardMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface gérant les stratégies de retour-arrière.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IBACKWARDMNG_H
#define ARCANE_CORE_IBACKWARDMNG_H
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
 * \internal
 * \brief Interface gérant les stratégies de retour-arrière.
 *
 * Cette interface est utilisée par le ITimeLoopMng pour gérer les
 * retour-arrière. Le principe du retour-arrière est de sauvegarder à une
 * itération donnée les valeurs des variables pour pouvoir revenir à
 * cette itération, par exemple en cas de problème dans le calcul.
 *
 * Il est possible de positionner une instance spécifique via
 * ITimeLoopMng::setBackwardMng();
 *
 * L'enchainement des opérations, effectué à la fin de chaque itération,
 * est géré par l'instance de ITimeLoopMng. Il est comme suit:
 *
 * \code
 * IBackwardMng bw = ...;
 * bw->beginAction();
 * if (bw->checkAndApplyRestore()){
 *   // Exécution des points d'entrée de restoration.
 * }
 * bw->checkAndApplySave();
 * bw->endAction();
 * \endcode
 */
class ARCANE_CORE_EXPORT IBackwardMng
{
 public:

  // Actions à entreprendre
  enum eAction
  {
    //! Sauvegarde
    Save,
    //! Restauration
    Restore
  };

 public:

  virtual ~IBackwardMng() = default;

 public:

  //! Initialisation du manager de retour en arrière
  virtual void init() = 0;

  //! Indique qu'on commence les actions de sauvegarde/restauration sont terminées
  virtual void beginAction() = 0;

  /*!
   * \brief Vérifie et applique la restauration si nécessaire.
   * \retval true si une restauration est effectuée.
   */
  virtual bool checkAndApplyRestore() = 0;

  /*!
   * \brief Vérifie et applique la sauvegarde des variables si nécessaire.
   * Si \a is_forced est vrai, force la sauvegarde.
   * \retval true si une sauvegarde est effectuée.
   */
  virtual bool checkAndApplySave(bool is_forced) = 0;

  //! Indique que les actions de sauvegarde/restauration sont terminées
  virtual void endAction() = 0;

  // Période de sauvegarde
  virtual void setSavePeriod(Integer n) = 0;

  // Récupère la période de sauvegarde
  virtual Integer savePeriod() const = 0;

  /*!
   * \brief Signale qu'on souhaite effectué un retour arrière.
   *
   * Le retour arrière aura lieu lors de l'appel à checkAndApplyRestore().
   *
   * En général il ne faut pas appeler directement cette méthode mais
   * plutôt ITimeLoopMng::goBackward().
   *
   * Depuis l'appel à cette méthode jusqu'à l'action effective du
   * retour-arrière lors de l'appel à checkAndApplyRestore(),
   * isBackwardEnabled() retourne \a true.
   */
  virtual void goBackward() = 0;

  /*!
   * \brief Indique si les sauvegardes de retour-arrière sont verrouillées.
   *
   * isLocked() est vrai s'il n'est pas possible de faire une
   * sauvegarde. C'est le cas par exemple lorsqu'on a effectué à l'itération
   * \a M un retour arrière vers l'itération N et qu'on n'est pas encore
   * revenu à l'itération \a M.
   */
  virtual bool isLocked() const = 0;

  /*!
   * \brief Indique si un retour-arrière est programmé.
   * \sa goBackward().
   */
  virtual bool isBackwardEnabled() const = 0;

  /*!
   * \brief Supprime les ressources associées au retour-arrière.
   *
   * Cette méthode est appelé pour désallouer les ressources
   * comme les sauvegardes des variables. Cette méthode est appelée
   * entre autre avant un équilibrage de charge puisqu'il ne sera
   * pas possible de faire un retour-arrière avant cet équilibrage.
   */
  virtual void clear() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

