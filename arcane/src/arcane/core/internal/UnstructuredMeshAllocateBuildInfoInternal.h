// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshAllocateBuildInfoInternal.h                 (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage non structuré.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_UNSTRUCTUREDMESHALLOCATEBUILDINFOINTERNAL_H
#define ARCANE_CORE_UNSTRUCTUREDMESHALLOCATEBUILDINFOINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/UnstructuredMeshAllocateBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Partie interne de UnstructuredMeshAllocateBuildInfo
 */
class ARCANE_CORE_EXPORT UnstructuredMeshAllocateBuildInfoInternal
{
  friend class UnstructuredMeshAllocateBuildInfo::Impl;

 public:

  ConstArrayView<Int64> cellsInfos() const;
  Int32 meshDimension() const;
  Int32 nbCell() const;

 private:

  UnstructuredMeshAllocateBuildInfo::Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
