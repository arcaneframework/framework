// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemInternalSortFunction.h                                 (C) 2000-2025 */
/*                                                                           */
/* Interface d'une fonction de tri des entités.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMINTERNALSORTFUNCTION_H
#define ARCANE_CORE_IITEMINTERNALSORTFUNCTION_H
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
 * \ingroup Mesh
 * \brief Interface d'une fonction de tri des entités.
 *
 * Cette classe est utilisée pour trier des entités.
 * Cela se fait lors de l'appel à la méthode sortItems().
 *
 * Pour simplifier le tri, il est préférable d'utiliser la
 * classe ItemInternalSortFunction en spécifiant la fonction
 * de comparaison.
 *
 */
class IItemInternalSortFunction
{
 public:

  virtual ~IItemInternalSortFunction() = default; //!< Libère les ressources

 public:

  /*!
   * \brief Nom de la fonction de tri.
   *
   * Les noms commençant par 'Arcane' sont réservés et ne doivent pas être
   * utilisés.
   */
  virtual const String& name() const = 0;

  /*!
   * \brief Trie les entités du tableau \a items.
   */
  virtual void sortItems(ItemInternalMutableArrayView items) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
