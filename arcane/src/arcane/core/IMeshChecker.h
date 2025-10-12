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

  /*!
   * \brief Indique si on active la vérification des propriétaires des entités.
   *
   * Cette vérification est effectuée lors de l'appel à checkValidConnectivity().
   * Si elle est active, on vérifie que les noeuds, arêtes et
   * faces ont bien le même propriétaire qu'une des mailles auxquels ils sont
   * connectés.
   *
   * C'est toujours le cas si lorsque les propriétaires sont gérés par %Arcane
   * et il est donc préférable de toujours faire cette vérification pour
   * garantir la cohérence des informations en parallèle. Cependant, si la
   * gestion des propriétaires est faite par l'utilisateur, il est possible
   * de désactiver cette vérification.
   */
  virtual void setIsCheckItemsOwner(bool v) = 0;

  //! Indique si la vérification des propriétaires des entités (vrai par défaut)
  virtual bool isCheckItemsOwner() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
