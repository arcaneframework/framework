// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemInternalSortFunction.h                                 (C) 2000-2008 */
/*                                                                           */
/* Interface d'une fonction de tri des entités.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IITEMINTERNALSORTFUNCTION_H
#define ARCANE_IITEMINTERNALSORTFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

  virtual ~IItemInternalSortFunction() {} //<! Libère les ressources

 public:

  /*!
   * \brief Nom de la fonction de tri.
   *
   * Les noms commençant par 'Arcane' sont réservés et ne doivent pas être
   * utilisés.
   */
  virtual const String& name() const =0;
  
  /*!
   * \brief Trie les entités du tableau \a items.
   */
  virtual void sortItems(ItemInternalMutableArrayView items) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
