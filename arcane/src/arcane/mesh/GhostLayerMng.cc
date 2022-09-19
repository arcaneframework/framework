// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerMng.cc                                            (C) 2000-2022 */
/*                                                                           */
/* Gestionnaire de couche fantômes d'un maillage.                            */
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
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_NB_GHOSTLAYER",true))
    m_nb_ghost_layer = std::clamp(v.value(),1,256);

  _initBuilderVersion();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerMng::
setNbGhostLayer(Integer n)
{
  if (n<0)
    ARCANE_THROW(ArgumentException,"Bad number of ghost layer '{0}'<0",n);
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
  if (n<2 || n>4)
    ARCANE_THROW(ArgumentException,"Bad value for builder version '{0}'. valid values are 2, 3 or 4.",n);
  m_builder_version = n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostLayerMng::
_initBuilderVersion()
{
  // La version par défaut est la 2.
  // La version 1 n'existe plus.
  // La version 3 est opérationnelle et plus extensible que la 2.
  // La version 4 est comme la version 3 mais permet d'être appelée
  // alors qu'il y a déjà des couches de mailles fantômes.
  // Si OK pour IFP, il faudra passer la version par défaut à la 3 ou 4. Il
  // reste cependant à traiter le cas des maillages AMR
  Integer default_version = 2;
  Integer version = default_version;
  String version_str = platform::getEnvironmentVariable("ARCANE_GHOSTLAYER_VERSION");
  if (!version_str.null()){
    if (builtInGetValue(version,version_str)){
      pwarning() << "Bad value for 'ARCANE_GHOSTLAYER_VERSION'";
    }
    if (version<2 || version>4)
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
