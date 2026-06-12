// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IGhostLayerMng.h                                            (C) 2000-2025 */
/*                                                                           */
/* Interface of the mesh ghost layer manager.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IGHOSTLAYERMNG_H
#define ARCANE_CORE_IGHOSTLAYERMNG_H
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
 * \internal
 * Interface of the mesh ghost layer manager.
 */
class IGhostLayerMng
{
 public:

  //! Releases resources
  virtual ~IGhostLayerMng() = default;

 public:

  //! Sets the number of ghost layers.
  virtual void setNbGhostLayer(Integer n) = 0;

  //! Number of ghost layers.
  virtual Integer nbGhostLayer() const = 0;

  /*!
   * \brief Sets the version of the ghost cell builder.
   * For now (version 3.3), the possible values are 2, 3, or 4.
   * The default value is 2. Values 3 and 4 allow support
   * of multiple ghost cell layers.
   */
  virtual void setBuilderVersion(Integer n) = 0;

  //! Ghost cell builder version.
  virtual Integer builderVersion() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
