// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAllocateBuildInfoInternal.h                    (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage cartésien.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CARTESIANMESHALLOCATEBUILDINFOINTERNAL_H
#define ARCANE_CORE_CARTESIANMESHALLOCATEBUILDINFOINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CartesianMeshAllocateBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne de CartesianMeshAllocateBuildInfo
 */
class ARCANE_CORE_EXPORT CartesianMeshAllocateBuildInfoInternal
{
  friend class CartesianMeshAllocateBuildInfo::Impl;

 public:

  ConstArrayView<Int64> cellsInfos() const;
  Int32 meshDimension() const;
  Int32 nbCell() const;

 private:

  CartesianMeshAllocateBuildInfo::Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
