// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshEventsImpl.h                                            (C) 2000-2023 */
/*                                                                           */
/* Implémentation des évènements sur le maillage.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_MESHEVENTS_H
#define ARCANE_MESH_MESHEVENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Event.h"

#include "arcane/core/MeshEvents.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation des évènements sur le maillage.
 */
class ARCANE_CORE_EXPORT MeshEventsImpl
{
 public:

  EventObservable<const MeshEventArgs&>& eventObservable(eMeshEventType type);

 private:

  EventObservable<const MeshEventArgs&> m_on_begin_prepare_dump;
  EventObservable<const MeshEventArgs&> m_on_end_prepare_dump;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
