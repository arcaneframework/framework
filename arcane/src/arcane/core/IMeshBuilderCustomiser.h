// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshBuilderCustomiser.h                                    (C) 2000-2026 */
/*                                                                           */
/* Interface of a mesh builder customisation service.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHBUILDERCUSTOMISER_H
#define ARCANE_CORE_IMESHBUILDERCUSTOMISER_H
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
 * \brief Interface of a mesh builder customisation service.
 */
class ARCANE_CORE_EXPORT IMeshBuilderCustomiser
{
 public:

  virtual ~IMeshBuilderCustomiser() = default; //<! Releases resources

 public:

  /*!
   * \brief Modify \a build_info to customise mesh creation.
   *
   * The implementation is allowed to change values of \a build_info,
   * like the `IParallelMng` or mesh factory used.
   */
  virtual void applyCustomisation(MeshBuildInfo& build_info) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
