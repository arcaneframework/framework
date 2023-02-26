// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshAllocateBuildInfo.h                         (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage non structuré.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UNSTRUCTUREDMESHALLOCATEBUILDINFO_H
#define ARCANE_UNSTRUCTUREDMESHALLOCATEBUILDINFO_H
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
 * \brief Informations pour allouer les entités d'un maillage non structuré.
 */
class ARCANE_CORE_EXPORT UnstructuredMeshAllocateBuildInfo
{
  class Impl;

 public:

  UnstructuredMeshAllocateBuildInfo(IPrimaryMesh* mesh);
  ~UnstructuredMeshAllocateBuildInfo();

 public:

  UnstructuredMeshAllocateBuildInfo(UnstructuredMeshAllocateBuildInfo&& from) = delete;
  UnstructuredMeshAllocateBuildInfo(const UnstructuredMeshAllocateBuildInfo& from) = delete;
  UnstructuredMeshAllocateBuildInfo& operator=(UnstructuredMeshAllocateBuildInfo&& from) = delete;
  UnstructuredMeshAllocateBuildInfo& operator=(const UnstructuredMeshAllocateBuildInfo& from) = delete;

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

