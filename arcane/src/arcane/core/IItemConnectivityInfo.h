// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IItemConnectivityInfo.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface for connectivity information by entity type.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IITEMCONNECTIVITYINFO_H
#define ARCANE_CORE_IITEMCONNECTIVITYINFO_H
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
 *
 * \brief Interface for connectivity information by entity type.
 *
 * This interface allows knowing for a given entity type
 * the maximum number of connected entities. This can be used
 * for example to size variables.
 *
 * Instances of this interface are generally retrieved
 * via IItemFamily::localConnectivityInfos() for local information within the
 * subdomain or IItemFamily::globalConnectivityInfos() for global information
 * across all meshes.
 */
class IItemConnectivityInfo
{
 public:

  virtual ~IItemConnectivityInfo() = default; //<! Releases resources

 public:

  //! Maximum number of nodes per entity
  virtual Integer maxNodePerItem() const =0;
  
  //! Maximum number of edges per entity
  virtual Integer maxEdgePerItem() const =0;

  //! Maximum number of faces per entity
  virtual Integer maxFacePerItem() const =0;

  //! Maximum number of cells per entity
  virtual Integer maxCellPerItem() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
