// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemFamilySerializerMngInternal.h                          (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des outils de sérialisation/désérialisation d'une famille.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMFAMILYSERIALIZERMNGINTERNAL_H
#define ARCANE_CORE_IITEMFAMILYSERIALIZERMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IItemFamily;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère la sérialisation/désérialisation des entités d'une famille.
 */
class ARCANE_CORE_EXPORT IItemFamilySerializerMngInternal
{
 public:

  virtual ~IItemFamilySerializerMngInternal() = default;

 public:

   /*!
   * \brief Finalise les allocations réalisées par les serializers enregistrés dans le gestionnaire.
   *
   * Utilisé pour le maillage polyédrique où les allocations ne sont réalisées qu'après avoir
   * effectué toutes les sérialisations pour toutes les familles
   */
  virtual void finalizeItemAllocation() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

