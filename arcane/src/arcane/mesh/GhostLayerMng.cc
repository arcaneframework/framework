// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GhostLayerMng.cc                                            (C) 2000-2013 */
/*                                                                           */
/* Gestionnaire de couche fantômes d'un maillage.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/mesh/GhostLayerMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GhostLayerMng::
GhostLayerMng(ITraceMng* tm)
: TraceAccessor(tm)
, m_nb_ghost_layer(1)
, m_builder_version(2)
{
  String nb_ghost_str = platform::getEnvironmentVariable("ARCANE_NB_GHOSTLAYER");
  Integer nb_ghost = 1;
  if (!nb_ghost_str.null())
    builtInGetValue(nb_ghost,nb_ghost_str);
  if (nb_ghost<=0)
    nb_ghost = 1;
  m_nb_ghost_layer = nb_ghost;

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
  if (n<1 || n>3)
    ARCANE_THROW(ArgumentException,"Bad value for builder version '{0}'. valid values are 2 or 3.",n);
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
  // Si OK pour IFP, il faudra passer la version par défaut à la 3. Il
  // reste cependant à traiter le cas des maillages AMR
  Integer default_version = 2;
  Integer version = default_version;
  String version_str = platform::getEnvironmentVariable("ARCANE_GHOSTLAYER_VERSION");
  if (!version_str.null()){
    if (builtInGetValue(version,version_str)){
      pwarning() << "Bad value for 'ARCANE_GHOSTLAYER_VERSION'";
    }
    if (version<1 || version>3)
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

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
