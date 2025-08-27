// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshInternal.h                                             (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de IMesh.                                         */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CORE_INTERNAL_IPOLYHEDRALMESHMODIFIER_H
#define ARCANE_CORE_INTERNAL_IPOLYHEDRALMESHMODIFIER_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IItemConnectivityMng;
class IPolyhedralMeshModifier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de IMesh.
 */
class ARCANE_CORE_EXPORT IPolyhedralMeshModifier
{
 public:
  virtual ~IPolyhedralMeshModifier() = default;

  virtual void addItems(Int64ConstArrayView unique_ids, Int32ArrayView local_ids, eItemKind ik, const String& family_name) = 0;
  virtual void removeItems(Int32ConstArrayView local_ids, eItemKind ik, const String& family_name) = 0;
};

}// namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif