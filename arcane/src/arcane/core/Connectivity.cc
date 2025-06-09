// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMesh.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Classe de description de la connectivité du maillage.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Connectivity.h"
#include "arcane/core/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Connectivity::
Connectivity(VariableScalarInteger connectivity)
: m_connectivity(connectivity)
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Connectivity::
enableConnectivity(const Integer c)
{
  _checkValid(c);
  _enableConnectivity(c);
}

/*---------------------------------------------------------------------------*/

void Connectivity::
disableConnectivity(const Integer c)
{
  _checkValid(c);
  _disableConnectivity(c);
}

/*---------------------------------------------------------------------------*/

bool Connectivity::
hasConnectivity(const Integer c) const
{
  _checkValid(c);
  return _hasConnectivity(m_connectivity(), c);
}

/*---------------------------------------------------------------------------*/

bool Connectivity::
isFrozen() const
{
  return _hasConnectivity(m_connectivity(), CT_Frozen);
}

/*---------------------------------------------------------------------------*/

void Connectivity::
freeze(IMesh* mesh)
{
  const Integer dim = mesh->dimension();

  switch (dim) {
  case 1:
    _enableConnectivity(CT_Default1D);
    _disableConnectivity(CT_EdgeConnectivity);
    _enableConnectivity(CT_Dim1D);
    break;
  case 2:
    _enableConnectivity(CT_Default2D);
    _disableConnectivity(CT_EdgeConnectivity);
    _enableConnectivity(CT_Dim2D);
    break;
  case 3:
    _enableConnectivity(CT_Default3D);
    _enableConnectivity(CT_Dim3D);
    break;
  default:
    ARCANE_FATAL("Mesh ({0}) dimension must be set before frezing connectivity (current={1})", mesh->name(), dim);
  }

  if (_hasConnectivity(m_connectivity(), CT_Dim2D) && hasConnectivity(m_connectivity(), CT_Dim3D))
    ARCANE_FATAL("A mesh cannot have both dimensions 2 and 3");

  // Connectivité minimale
  // c'est la relation avec une cellule qui définit le non-rejet d'un item
  if (hasConnectivity(CT_HasNode))
    _enableConnectivity(CT_CellToNode);
  if (hasConnectivity(CT_HasEdge))
    _enableConnectivity(CT_CellToEdge + CT_NodeToEdge + CT_EdgeToNode + CT_EdgeToCell); // il faudrait faire avec moins
  if (hasConnectivity(CT_HasFace))
    _enableConnectivity(CT_CellToFace);

  _enableConnectivity(CT_Frozen);
}

/*---------------------------------------------------------------------------*/

Integer Connectivity::
getPrealloc(const Integer connectivity, eItemKind kindA, eItemKind kindB)
{
  const Integer x = -1; // illegal value

  Integer dim = 0;
  if (hasConnectivity(connectivity, CT_Dim1D))
    dim = 1;
  if (hasConnectivity(connectivity, CT_Dim2D)) {
    if (dim != 0) {
      throw FatalErrorException(A_FUNCINFO, "A mesh cannot have both dimensions 1 and 2");
    }
    else {
      dim = 2;
    }
  }
  if (hasConnectivity(connectivity, CT_Dim3D)) {
    if (dim != 0) {
      throw FatalErrorException(A_FUNCINFO, "A mesh cannot have both dimensions (1 or 2) and 3");
    }
    else {
      dim = 3;
    }
  }

  if (kindA > NB_ITEM_KIND || kindB > NB_ITEM_KIND)
    throw FatalErrorException(A_FUNCINFO, String::format("Invalid connectivity kind ({0}To{1})", kindA, kindB));

  Integer prealloc = 0;
  switch (dim) {
  case 1: {
    // utilise l'ordre de l'enum eItemKind (contient tous les kind jusqu'à Particle)
    // (kindA=ligne,kindB=colonne) => nb d'item de type 'itemB' pour un kind de type 'kindA'
    Integer preallocs[8][8] = { { x, x, 2, 2, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { 2, x, x, 2, x, x, x, x },
                                { 2, x, 2, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x } };
    prealloc = preallocs[kindA][kindB];
  } break;
  case 2: {
    // utilise l'ordre de l'enum eItemKind (contient tous les kind jusqu'à Particle)
    // (kindA=ligne,kindB=colonne) => nb d'item de type 'itemB' pour un kind de type 'kindA'
    Integer preallocs[8][8] = { { x, x, 4, 4, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { 4, x, x, 2, x, x, x, x },
                                { 4, x, 4, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x } };
    prealloc = preallocs[kindA][kindB];
  } break;
  case 3: {
    // utilise l'ordre de l'enum eItemKind (contient tous les kind jusqu'à Particle)
    // (kindA=ligne,kindB=colonne) => nb d'item de type 'itemB' pour un kind de type 'kindA'
    Integer preallocs[8][8] = { { x, 6, 12, 8, x, x, x, x },
                                { 2, x, 4, 4, x, x, x, x },
                                { 4, 4, x, 2, x, x, x, x },
                                { 8, 12, 6, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x },
                                { x, x, x, x, x, x, x, x } };
    prealloc = preallocs[kindA][kindB];
  } break;
  default:
    throw FatalErrorException(A_FUNCINFO, String::format("Not supported mesh dimension (current={0}): must be 1, 2 or 3", dim));
  }

  if (prealloc == x)
    throw FatalErrorException(A_FUNCINFO, String::format("Cannot request prealloc of {1} for {0} family", kindA, kindB));

  Integer kindAkindBConnectivity = kindsToConnectivity(kindA, kindB);
  if (hasConnectivity(connectivity, kindAkindBConnectivity)) {
    return prealloc;
  }
  else {
    return 0;
  }
}

/*---------------------------------------------------------------------------*/

void Connectivity::
print(std::ostream& o, const Integer connectivity)
{
  Integer position = 0;
  //   if (hasConnectivity(connectivity,CT_NodeToNode))
  //     o << ((position++)?"|":"") << "NodeToNode";
  if (hasConnectivity(connectivity, CT_NodeToEdge))
    o << ((position++) ? "|" : "") << "NodeToEdge";
  if (hasConnectivity(connectivity, CT_NodeToFace))
    o << ((position++) ? "|" : "") << "NodeToFace";
  if (hasConnectivity(connectivity, CT_NodeToCell))
    o << ((position++) ? "|" : "") << "NodeToCell";
  if (hasConnectivity(connectivity, CT_EdgeToNode))
    o << ((position++) ? "|" : "") << "EdgeToNode";
  //   if (hasConnectivity(connectivity,CT_EdgeToEdge))
  //     o << ((position++)?"|":"") << "EdgeToEdge";
  if (hasConnectivity(connectivity, CT_EdgeToFace))
    o << ((position++) ? "|" : "") << "EdgeToFace";
  if (hasConnectivity(connectivity, CT_EdgeToCell))
    o << ((position++) ? "|" : "") << "EdgeToCell";
  if (hasConnectivity(connectivity, CT_FaceToNode))
    o << ((position++) ? "|" : "") << "FaceToNode";
  if (hasConnectivity(connectivity, CT_FaceToEdge))
    o << ((position++) ? "|" : "") << "FaceToEdge";
  if (hasConnectivity(connectivity, CT_FaceToFace))
    o << ((position++) ? "|" : "") << "FaceToFace";
  if (hasConnectivity(connectivity, CT_FaceToCell))
    o << ((position++) ? "|" : "") << "FaceToCell";
  if (hasConnectivity(connectivity, CT_CellToNode))
    o << ((position++) ? "|" : "") << "CellToNode";
  if (hasConnectivity(connectivity, CT_CellToEdge))
    o << ((position++) ? "|" : "") << "CellToEdge";
  if (hasConnectivity(connectivity, CT_CellToFace))
    o << ((position++) ? "|" : "") << "CellToFace";
  //   if (hasConnectivity(connectivity,CT_CellToCell))
  //     o << ((position++)?"|":"") << "CellToCell";
  if (hasConnectivity(connectivity, CT_HasNode))
    o << ((position++) ? "|" : "") << "HasNode";
  if (hasConnectivity(connectivity, CT_HasEdge))
    o << ((position++) ? "|" : "") << "HasEdge";
  if (hasConnectivity(connectivity, CT_HasFace))
    o << ((position++) ? "|" : "") << "HasFace";
  if (hasConnectivity(connectivity, CT_HasCell))
    o << ((position++) ? "|" : "") << "HasCell";
  if (hasConnectivity(connectivity, CT_Frozen))
    o << ((position++) ? "|" : "") << "Frozen";
  //   if (hasConnectivity(connectivity,CT_Dim1D))
  //     o << ((position++)?"|":"") << "Dim1D";
  if (hasConnectivity(connectivity, CT_Dim2D))
    o << ((position++) ? "|" : "") << "Dim2D";
  if (hasConnectivity(connectivity, CT_Dim3D))
    o << ((position++) ? "|" : "") << "Dim3D";

  if (position == 0)
    o << "Null";
}

/*---------------------------------------------------------------------------*/

void Connectivity::Printer::print(std::ostream& o) const
{
  Connectivity::print(o, m_connectivity);
}

/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o, const Connectivity::Printer& p)
{
  p.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Connectivity::
_enableConnectivity(const Integer c)
{
  _checkFrozen();
  m_connectivity = m_connectivity() | c;
}

/*---------------------------------------------------------------------------*/

void Connectivity::
_disableConnectivity(const Integer c)
{
  _checkFrozen();
  m_connectivity = m_connectivity() & ~c;
}

/*---------------------------------------------------------------------------*/

void Connectivity::
_checkValid(const Integer c)
{
  if ((c & ~CT_FullConnectivity3D) && (c & ~CT_GraphConnectivity))
    ARCANE_FATAL("Illegal connectivity flag");
}

/*---------------------------------------------------------------------------*/

void Connectivity::
_checkFrozen() const
{
  if (m_connectivity() & CT_Frozen)
    ARCANE_FATAL("Cannot modify frozen connectivity");
}

/*---------------------------------------------------------------------------*/

Integer Connectivity::
kindsToConnectivity(eItemKind kindA, eItemKind kindB)
{
  switch (kindA) {
  case IK_Node:
  case IK_Edge:
  case IK_Face:
  case IK_Cell:
  case IK_DoF:
  case IK_Particle:
    break;
  default:
    throw FatalErrorException(A_FUNCINFO, String::format("Connectivity from kind {0} not supported", kindA));
  }
  switch (kindB) {
  case IK_Node:
  case IK_Edge:
  case IK_Face:
  case IK_Cell:
  case IK_DoF:
  case IK_Particle:
    break;
  default:
    ARCANE_FATAL("Connectivity to kind {0} not supported", kindB);
  }
  if (kindA == IK_DoF) {
    switch (kindB) {
    case IK_Node:
      return CT_DoFToNode;
    case IK_Edge:
      return CT_DoFToEdge;
    case IK_Face:
      return CT_DoFToFace;
    case IK_Cell:
      return CT_DoFToCell;
    case IK_DoF:
      return CT_DoFToDoF;
    case IK_Particle:
      return CT_DoFToParticle;
    default:
      ARCANE_FATAL("Connectivity to kind {0} not supported", kindB);
    }
  }
  else
    return 1 << (4 * kindA + kindB + 1);
  ARCANE_FATAL("Connectivity {0} to kind {1} is not supported", kindA, kindB);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
