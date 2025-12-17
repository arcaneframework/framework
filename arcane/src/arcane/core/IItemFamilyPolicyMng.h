// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilyPolicyMng.h                                      (C) 2000-2025 */
/*                                                                           */
/* Politiques d'une famille d'entités.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYPOLICYMNG_H
#define ARCANE_CORE_IITEMFAMILYPOLICYMNG_H
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
 * \brief Interface des politiques d'une famille d'entités.
 */
class ARCANE_CORE_EXPORT IItemFamilyPolicyMng
{
 public:

  virtual ~IItemFamilyPolicyMng() = default;

 public:

  //! Politique de compactage
  virtual IItemFamilyCompactPolicy* compactPolicy() = 0;
  /*!
   * \brief Créé une instance pour l'échange d'entités entre sous-domaines.
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  virtual IItemFamilyExchanger* createExchanger() = 0;

  /*!
   * \brief Créé une instance pour la sérialisation des entités.
   * L'instance retournée doit être détruite par l'opérateur delete.
   *
   * \a with_flags indique si on doit sérialiser la valeur de Item::flags().
   * Cela n'est pas forcément supporté pour toutes les familles.
   */
  virtual IItemFamilySerializer* createSerializer(bool with_flags = false) = 0;

  /*!
   * \brief Ajoute une fabrique pour une étape de la sérialisation.
   *
   * \a factory reste la propriété de l'appelant et ne doit pas être détruit
   * tant que cette instance existe.
   */
  virtual void addSerializeStep(IItemFamilySerializeStepFactory* factory) = 0;

  //! Supprime une fabrique pour une étape de la sérialisation.
  virtual void removeSerializeStep(IItemFamilySerializeStepFactory* factory) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
