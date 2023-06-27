// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEvents.h                                                (C) 2000-2023 */
/*                                                                           */
/* Evènements sur un maillage.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHEVENTS_H
#define ARCANE_CORE_MESHEVENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Evènements générés par IMesh
enum class eMeshEventType
{
  //! Evènement envoyé au début de prepareForDump()
  BeginPrepareDump,
  //! Evènement envoyé à la fin de prepareForDump()
  EndPrepareDump
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments des évènements sur le maillage.
 */
class ARCANE_CORE_EXPORT MeshEventArgs
{
 public:

  MeshEventArgs(IMesh* mesh, eMeshEventType type)
  : m_mesh(mesh)
  , m_type(type)
  {}

 public:

  IMesh* mesh() const { return m_mesh; }
  eMeshEventType type() const { return m_type; }

 private:

  IMesh* m_mesh = nullptr;
  eMeshEventType m_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
