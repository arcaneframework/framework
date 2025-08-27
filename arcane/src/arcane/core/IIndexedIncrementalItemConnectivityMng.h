// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIndexedIncrementalItemConnectivityMng.h                    (C) 2000-2022 */
/*                                                                           */
/* Interface du gestionnaire de 'IIndexedIncrementalItemConnectivity'.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IINDEXEDINCREMENTALITEMCONNECTIVITYMNG_H
#define ARCANE_IINDEXEDINCREMENTALITEMCONNECTIVITYMNG_H
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
 * \brief Interface du gestionnaire des connectivités incrémentales indexées sur les entités.
 */
class ARCANE_CORE_EXPORT IIndexedIncrementalItemConnectivityMng
{
 public:

  virtual ~IIndexedIncrementalItemConnectivityMng() = default;

 public:

  /*!
   * \brief Cherche ou créé une connectivité.
   *
   * Lève une exception si une connectivité de nom \a name existe déjà mais
   * pas avec le même couple (source,target).
   * L'instance reste propriétaire de la connectivité retournée.
   */
  virtual Ref<IIndexedIncrementalItemConnectivity>
  findOrCreateConnectivity(IItemFamily* source, IItemFamily* target, const String& name) = 0;

  /*!
   * \brief Cherche ou créé une connectivité.
   *
   * Lève une exception si la connectivité de nom \a name n'est pas trouvée.
   * L'instance reste propriétaire de la connectivité retournée.
   */
  virtual Ref<IIndexedIncrementalItemConnectivity>
  findConnectivity(const String& name) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
