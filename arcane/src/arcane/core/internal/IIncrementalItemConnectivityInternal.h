// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IIncrementalItemConnectivityInternal.h                      (C) 2000-2024 */
/*                                                                           */
/* API interne à Arcane de IncrementalItemConnectivity                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IINCREMENTALITEMCONNECTIVITYINTERNAL_H
#define ARCANE_CORE_INTERNAL_IINCREMENTALITEMCONNECTIVITYINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/IItemConnectivityAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur l'utilisation mémoire pour les connectivités.
 */
class ItemConnectivityMemoryInfo
{
 public:

  //! Nombre total de Int32 utilisés (correspoind à la somme des size())
  Int64 m_total_size = 0;
  //! Nombre total de Int32 allouées (correspond à la somme des capacity())
  Int64 m_total_capacity = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief API interne à Arcane de IIncrementalItemConnectivity.
 */
class ARCANE_CORE_EXPORT IIncrementalItemConnectivityInternal
{
 public:

  virtual ~IIncrementalItemConnectivityInternal() = default;

 public:

  //! Réduit au minimum l'utilisation mémoire pour les connectivités
  virtual void shrinkMemory() = 0;

  //! Ajoute \a mem_info les informations mémoire de l'instance.
  virtual void addMemoryInfos(ItemConnectivityMemoryInfo& mem_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
