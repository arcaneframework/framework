// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerMng.cc                                            (C) 2000-2022 */
/*                                                                           */
/* Mesh ghost layer manager.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/mesh/GhostLayerMng.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerMng::
GhostLayerMng(ITraceMng* tm)
: TraceAccessor(tm)
, m_nb_ghost_layer(1)
, m_builder_version(3)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_NB_GHOSTLAYER", true))
    m_nb_ghost_layer = std::clamp(v.value(), 1, 256);

  _initBuilderVersion();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerMng::
setNbGhostLayer(Integer n)
{
  if (n < 0)
    ARCANE_THROW(ArgumentException, "Bad number of ghost layer '{0}'<0", n);
  m_nb_ghost_layer = n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer GhostLayerMng::
nbGhostLayer() const
{
  return m_nb_ghost_layer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerMng::
setBuilderVersion(Integer n)
{
  if (n < 2 || n > 4)
    ARCANE_THROW(ArgumentException, "Bad value for builder version '{0}'. valid values are 2, 3 or 4.", n);
  m_builder_version = n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerMng::
_initBuilderVersion()
{
  // The default version is 2.
  // Version 1 no longer exists.
  // Version 3 is operational and more extensible than 2.
  // Version 4 is like version 3 but allows being called
  // even if there are already ghost cell layers.
  // If OK for IFP, the default version should be set to 3 or 4. However,
  // the case of AMR meshes still needs to be handled
  Integer default_version = 2;
  Integer version = default_version;
  String version_str = platform::getEnvironmentVariable("ARCANE_GHOSTLAYER_VERSION");
  if (!version_str.null()) {
    if (builtInGetValue(version, version_str)) {
      pwarning() << "Bad value for 'ARCANE_GHOSTLAYER_VERSION'";
    }
    if (version < 2 || version > 4)
      version = default_version;
  }
  m_builder_version = version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer GhostLayerMng::
builderVersion() const
{
  return m_builder_version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
