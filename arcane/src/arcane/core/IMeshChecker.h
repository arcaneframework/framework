// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshChecker.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface de méthodes de vérification d'un maillage.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHCHECKER_H
#define ARCANE_CORE_IMESHCHECKER_H
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
 * \brief Interface de méthodes de vérification d'un maillage.
 */
class IMeshChecker
{
 public:

  virtual ~IMeshChecker() = default; //!< Libère les ressources

 public:

  //! Maillage associé
  virtual IMesh* mesh() = 0;

  /*!
   * \brief Positionne le niveau de vérification du maillage.
   *
   * 0 - tests désactivés
   * 1 - tests partiels, après les endUpdate()
   * 2 - tests complets, après les endUpdate()
   */
  virtual void setCheckLevel(Integer level) = 0;

  //! Niveau actuel de vérification
  virtual Integer checkLevel() const = 0;

  /*!
   * \brief Vérification de la validité des structures internes de maillage (interne).
   */
  virtual void checkValidMesh() = 0;

  /*!
   * \brief Vérification de la validité du maillage.
   *
   * Il s'agit d'une vérification globale entre tous les sous-domaines.
   *
   * Elle vérifie notamment que la connectivité est cohérente entre
   * les sous-domaines.
   *
   * La vérification peut-être assez coûteuse en temps CPU.
   * Cette méthode est collective.
   */
  virtual void checkValidMeshFull() = 0;

  /*!
   * \brief Vérifie que les sous-domaines sont correctement répliqués.
   *
   * Les vérifications suivantes sont faites:
   * - mêmes familles d'entité et mêmes valeurs pour ces familles.
   * - mêmes coordonnées des noeuds du maillage.
   */
  virtual void checkValidReplication() = 0;

  /*!
   * \brief Vérifie la synchronisation des variables.
   *
   * Vérifie pour chaque variable que ses valeurs sur les entités fantômes sont
   * les mêmes que la valeur sur le sous-domaine propriétaire de l'entité.
   *
   * Les variables sur les particules ne sont pas comparées.
   *
   * Lève une exception FatalErrorException en cas d'erreur.
   */
  virtual void checkVariablesSynchronization() = 0;

  /*!
   * \brief Vérifie la synchronisation sur les groupes d'entités.
   *
   * Vérifie pour chaque groupe de chaque famille (autre que les particules)
   * que les entités sont les mêmes sur chaque sous-domaine.
   *
   * Lève une exception FatalErrorException en cas d'erreur.
   */
  virtual void checkItemGroupsSynchronization() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
