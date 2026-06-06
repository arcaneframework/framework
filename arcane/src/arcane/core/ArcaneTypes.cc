// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneTypes.cc                                              (C) 2000-2012 */
/*                                                                           */
/* Definition of general Arcane types.                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"

// The following .h files are only useful for exporting the symbols
// they contain. This is necessary under Windows and sometimes under
// unix depending on the chosen options
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ITransferValuesParallelOperation.h"
#include "arcane/core/IMeshFactory.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshReader.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT const char*
itemKindName(eItemKind kind)
{
  switch (kind) {
  case IK_Cell:
    return "Cell";
  case IK_Node:
    return "Node";
  case IK_Face:
    return "Face";
  case IK_Edge:
    return "Edge";
  case IK_Particle:
    return "Particle";
  case IK_DoF:
    return "DoF";
  case IK_Unknown:
    return "None";
  }
  return "Invalid";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type_id)
{
  switch (type_id) {
  case IT_DualNode:
    return IK_Node;
  case IT_DualEdge:
    return IK_Edge;
  case IT_DualFace:
    return IK_Face;
  case IT_DualCell:
    return IK_Cell;
  case IT_DualParticle:
    return IK_Particle;
  default:
    return IK_Unknown;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& ostr, eItemKind item_kind)
{
  ostr << itemKindName(item_kind);
  return ostr;
}

extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>>(std::istream& istr, eItemKind& item_kind)
{
  String buf;
  istr >> buf;
  if (buf == "Node") {
    item_kind = IK_Node;
  }
  else if (buf == "Edge") {
    item_kind = IK_Edge;
  }
  else if (buf == "Face") {
    item_kind = IK_Face;
  }
  else if (buf == "Cell") {
    item_kind = IK_Cell;
  }
  else if (buf == "Particle") {
    item_kind = IK_Particle;
  }
  else if (buf == "DoF") {
    item_kind = IK_DoF;
  }
  else if (buf == "None") {
    item_kind = IK_Unknown;
  }
  else
    istr.setstate(std::ios_base::failbit);
  return istr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_CORE_EXPORT const char*
timePhaseName(eTimePhase time_phase)
{
  switch (time_phase) {
  case TP_Computation:
    return "Computation";
  case TP_Communication:
    return "Communication";
  case TP_InputOutput:
    return "InputOutput";
  }
  return "(Invalid)";
}

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& ostr, eTimePhase time_phase)
{
  ostr << timePhaseName(time_phase);
  return ostr;
}

extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>>(std::istream& istr, eTimePhase& time_phase)
{
  String buf;
  istr >> buf;
  if (buf == "Computation") {
    time_phase = TP_Communication;
  }
  else if (buf == "Communication") {
    time_phase = TP_Communication;
  }
  else if (buf == "InputOutput") {
    time_phase = TP_InputOutput;
  }
  else
    istr.setstate(std::ios_base::failbit);
  return istr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshDirection md)
{
  if (md == MD_DirX)
    o << "DirX";
  else if (md == MD_DirY)
    o << "DirY";
  else if (md == MD_DirZ)
    o << "DirZ";
  else
    o << "DirUnknown";
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
