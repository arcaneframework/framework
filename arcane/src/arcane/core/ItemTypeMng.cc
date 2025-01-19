// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeMng.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire des types d'entite du maillage.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypeMng.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/IOException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/MultiBuffer.h"

#include "arcane/core/ItemTypeInfoBuilder.h"
#include "arcane/core/IParallelSuperMng.h"
#include "arcane/core/ItemTypeInfoBuilder.h"

// AMR
#include "arcane/ItemRefinementPattern.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer ItemTypeMng::m_nb_builtin_item_type = NB_BASIC_ITEM_TYPE;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeMng::
ItemTypeMng()
: m_initialized(false)
, m_types_buffer(0)
{
  m_initialized_counter = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeMng::
~ItemTypeMng()
{
  delete m_types_buffer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeMng::
build(IParallelSuperMng* parallel_mng, ITraceMng* trace)
{
  // Avec MPC, cette fonction peut être appelée plusieurs fois
  // dans des threads différents. Comme tous les threads partagent
  // le même singleton, seul le premier thread fait réelement l'init.
  // ATTENTION: Cela est incompatible avec le mode readTypes()
  // ou on lit les connectivités dans un fichier ARCANE_ITEM_TYPE_FILE.
  Int32 max_rank = parallel_mng->commSize() + 1;
  Int32 init_counter = ++m_initialized_counter;
  if (init_counter == 1) {
    _build(parallel_mng, trace);
    m_initialized = true;
    m_initialized_counter = max_rank;
  }
  else
    // Ceux qui ne font pas l'init doivent attendre que cette derniere
    // soit faite.
    while (init_counter < max_rank)
      init_counter = m_initialized_counter.load();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeMng::
_build(IParallelSuperMng* parallel_mng, ITraceMng* trace)
{
  // Construit la connectivité des éléments.
  // Pour les éléments classiques, la connectivité est la même que
  // celle de VTK, disponible dans le document:
  //
  // https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

  m_trace = trace;
  m_types.resize(m_nb_builtin_item_type);
  m_types_buffer = new MultiBufferT<ItemTypeInfoBuilder>();

  // Null
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_NullType] = type;

    type->setInfos(this, IT_NullType, "NullType", 0, 0, 0);
  }

  // Vertex
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Vertex] = type;

    type->setInfos(this, IT_Vertex, "Vertex", 0, 0, 0);
    // TODO regarder si ce type est autorisé pour les mailles.
    // Si ce n'est pas le cas, il faudrait définir un type
    // pour les mailles 0D qui sont assimilables à des points.
  }

  // FaceVertex (face pour les maillages 1D)
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_FaceVertex] = type;

    type->setInfos(this, IT_FaceVertex, "FaceVertex", 1, 0, 0);
    type->setIsValidForCell(false);
  }

  // Line2
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Line2] = type;

    type->setInfos(this, IT_Line2, "Line2", 2, 0, 0);
    type->setIsValidForCell(false);
  }

  // Line3
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Line3] = type;

    type->setInfos(this, IT_Line3, "Line3", 3, 0, 0);
    type->setIsValidForCell(false);
  }

  // CellLine2 (mailles pour les maillages 1D)
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_CellLine2] = type;

    type->setInfos(this, IT_CellLine2, "CellLine2", 2, 0, 2);

    type->addFaceVertex(0, 0);
    type->addFaceVertex(1, 1);
  }

  /**
   * SDP: Pour les polygones les faces et les aretes sont identiques.
   *
   * @note lors des declarations des aretes, on donne pour faces les
   * aretes qui sont jointes a l'arete courante
   */

  // Triangle3
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Triangle3] = type;

    type->setInfos(this, IT_Triangle3, "Triangle3", 3, 3, 3);

    type->addFaceLine(0, 0, 1);
    type->addFaceLine(1, 1, 2);
    type->addFaceLine(2, 2, 0);

    type->addEdge(0, 0, 1, 1, 2);
    type->addEdge(1, 1, 2, 2, 0);
    type->addEdge(2, 2, 0, 0, 1);
  }

  // Triangle6
  {
    // TODO: Pour l'instant comme triangle3
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Triangle6] = type;

    type->setInfos(this, IT_Triangle6, "Triangle6", 6, 3, 3);

    type->addFaceLine3(0, 0, 1, 3);
    type->addFaceLine3(1, 1, 2, 4);
    type->addFaceLine3(2, 2, 0, 5);

    type->addEdge(0, 0, 1, 1, 2);
    type->addEdge(1, 1, 2, 2, 0);
    type->addEdge(2, 2, 0, 0, 1);
  }

  // Quad4
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Quad4] = type;

    type->setInfos(this, IT_Quad4, "Quad4", 4, 4, 4);

    type->addFaceLine(0, 0, 1);
    type->addFaceLine(1, 1, 2);
    type->addFaceLine(2, 2, 3);
    type->addFaceLine(3, 3, 0);

    type->addEdge(0, 0, 1, 3, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 0, 2, 0);
  }

  // Quad8
  {
    // TODO: Pour l'instant comme quad4
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Quad8] = type;

    type->setInfos(this, IT_Quad8, "Quad8", 8, 4, 4);

    type->addFaceLine3(0, 0, 1, 4);
    type->addFaceLine3(1, 1, 2, 5);
    type->addFaceLine3(2, 2, 3, 6);
    type->addFaceLine3(3, 3, 0, 7);

    type->addEdge(0, 0, 1, 3, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 0, 2, 0);
  }

  // Pentagon5
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Pentagon5] = type;

    type->setInfos(this, IT_Pentagon5, "Pentagon5", 5, 5, 5);

    type->addFaceLine(0, 0, 1);
    type->addFaceLine(1, 1, 2);
    type->addFaceLine(2, 2, 3);
    type->addFaceLine(3, 3, 4);
    type->addFaceLine(4, 4, 0);

    type->addEdge(0, 0, 1, 4, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 4, 2, 4);
    type->addEdge(4, 4, 0, 3, 0);
  }

  // Hexagon6
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Hexagon6] = type;

    type->setInfos(this, IT_Hexagon6, "Hexagon6", 6, 6, 6);

    type->addFaceLine(0, 0, 1);
    type->addFaceLine(1, 1, 2);
    type->addFaceLine(2, 2, 3);
    type->addFaceLine(3, 3, 4);
    type->addFaceLine(4, 4, 5);
    type->addFaceLine(5, 5, 0);

    type->addEdge(0, 0, 1, 5, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 4, 2, 4);
    type->addEdge(4, 4, 5, 3, 5);
    type->addEdge(5, 5, 0, 4, 0);
  }

  // Hexaedron8
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Hexaedron8] = type;

    type->setInfos(this, IT_Hexaedron8, "Hexaedron8", 8, 12, 6);

    type->addFaceQuad(0, 0, 3, 2, 1);
    type->addFaceQuad(1, 0, 4, 7, 3);
    type->addFaceQuad(2, 0, 1, 5, 4);
    type->addFaceQuad(3, 4, 5, 6, 7);
    type->addFaceQuad(4, 1, 2, 6, 5);
    type->addFaceQuad(5, 2, 3, 7, 6);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 4, 0);
    type->addEdge(2, 2, 3, 5, 0);
    type->addEdge(3, 3, 0, 1, 0);
    type->addEdge(4, 0, 4, 1, 2);
    type->addEdge(5, 1, 5, 2, 4);
    type->addEdge(6, 2, 6, 4, 5);
    type->addEdge(7, 3, 7, 5, 1);
    type->addEdge(8, 4, 5, 3, 2);
    type->addEdge(9, 5, 6, 3, 4);
    type->addEdge(10, 6, 7, 3, 5);
    type->addEdge(11, 7, 4, 3, 1);
  }

  // Hexaedron20
  {
    // Pour l'instant comme Hexaedron8
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Hexaedron20] = type;

    type->setInfos(this, IT_Hexaedron20, "Hexaedron20", 20, 12, 6);

    type->addFaceQuad8(0, 0, 3, 2, 1, 11, 10, 9, 8);
    type->addFaceQuad8(1, 0, 4, 7, 3, 16, 15, 19, 11);
    type->addFaceQuad8(2, 0, 1, 5, 4, 8, 17, 12, 16);
    type->addFaceQuad8(3, 4, 5, 6, 7, 19, 13, 14, 15);
    type->addFaceQuad8(4, 1, 2, 6, 5, 9, 18, 13, 17);
    type->addFaceQuad8(5, 2, 3, 7, 6, 10, 19, 14, 18);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 4, 0);
    type->addEdge(2, 2, 3, 5, 0);
    type->addEdge(3, 3, 0, 1, 0);
    type->addEdge(4, 0, 4, 1, 2);
    type->addEdge(5, 1, 5, 2, 4);
    type->addEdge(6, 2, 6, 4, 5);
    type->addEdge(7, 3, 7, 5, 1);
    type->addEdge(8, 4, 5, 3, 2);
    type->addEdge(9, 5, 6, 3, 4);
    type->addEdge(10, 6, 7, 3, 5);
    type->addEdge(11, 7, 4, 3, 1);
  }

  // Pyramid5
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Pyramid5] = type;

    type->setInfos(this, IT_Pyramid5, "Pyramid5", 5, 8, 5);

    type->addFaceQuad(0, 0, 3, 2, 1);
    type->addFaceTriangle(1, 0, 4, 3);
    type->addFaceTriangle(2, 0, 1, 4);
    type->addFaceTriangle(3, 1, 2, 4);
    type->addFaceTriangle(4, 2, 3, 4);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 3, 4, 0);
    type->addEdge(3, 3, 0, 1, 0);
    type->addEdge(4, 0, 4, 1, 2);
    type->addEdge(5, 1, 4, 2, 3);
    type->addEdge(6, 2, 4, 3, 4);
    type->addEdge(7, 3, 4, 4, 1);
  }

  // Pentaedron6
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Pentaedron6] = type;

    type->setInfos(this, IT_Pentaedron6, "Pentaedron6", 6, 9, 5);

    type->addFaceTriangle(0, 0, 2, 1);
    type->addFaceQuad(1, 0, 3, 5, 2);
    type->addFaceQuad(2, 0, 1, 4, 3);
    type->addFaceTriangle(3, 3, 4, 5);
    type->addFaceQuad(4, 1, 2, 5, 4);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 4, 0);
    type->addEdge(2, 2, 0, 1, 0);
    type->addEdge(3, 0, 3, 1, 2);
    type->addEdge(4, 1, 4, 2, 4);
    type->addEdge(5, 2, 5, 4, 1);
    type->addEdge(6, 3, 4, 3, 2);
    type->addEdge(7, 4, 5, 3, 4);
    type->addEdge(8, 5, 3, 3, 1);
  }

  // Tetraedron4
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Tetraedron4] = type;

    type->setInfos(this, IT_Tetraedron4, "Tetraedron4", 4, 6, 4);

    type->addFaceTriangle(0, 0, 2, 1);
    type->addFaceTriangle(1, 0, 3, 2);
    type->addFaceTriangle(2, 0, 1, 3);
    type->addFaceTriangle(3, 1, 2, 3);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 0, 1, 0);
    type->addEdge(3, 0, 3, 1, 2);
    type->addEdge(4, 1, 3, 2, 3);
    type->addEdge(5, 2, 3, 3, 1);
  }

  // Tetraedron10
  {
    // Pour l'instant comme Tetraedron4
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Tetraedron10] = type;

    type->setInfos(this, IT_Tetraedron10, "Tetraedron10", 10, 6, 4);

    type->addFaceTriangle6(0, 0, 2, 1, 6, 5, 4);
    type->addFaceTriangle6(1, 0, 3, 2, 7, 9, 6);
    type->addFaceTriangle6(2, 0, 1, 3, 4, 8, 7);
    type->addFaceTriangle6(3, 1, 2, 3, 5, 9, 8);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 0, 1, 0);
    type->addEdge(3, 0, 3, 1, 2);
    type->addEdge(4, 1, 3, 2, 3);
    type->addEdge(5, 2, 3, 3, 1);
  }

  // Heptaedron10
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Heptaedron10] = type;

    type->setInfos(this, IT_Heptaedron10, "Heptaedron10", 10, 15, 7);

    type->addFacePentagon(0, 0, 4, 3, 2, 1);
    type->addFacePentagon(1, 5, 6, 7, 8, 9);
    type->addFaceQuad(2, 0, 1, 6, 5);
    type->addFaceQuad(3, 1, 2, 7, 6);
    type->addFaceQuad(4, 2, 3, 8, 7);
    type->addFaceQuad(5, 3, 4, 9, 8);
    type->addFaceQuad(6, 4, 0, 5, 9);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 3, 4, 0);
    type->addEdge(3, 3, 4, 5, 0);
    type->addEdge(4, 4, 0, 6, 0);
    type->addEdge(5, 5, 6, 1, 2);
    type->addEdge(6, 6, 7, 1, 3);
    type->addEdge(7, 7, 8, 1, 4);
    type->addEdge(8, 8, 9, 1, 5);
    type->addEdge(9, 9, 5, 1, 6);
    type->addEdge(10, 0, 5, 6, 2);
    type->addEdge(11, 1, 6, 2, 3);
    type->addEdge(12, 2, 7, 3, 4);
    type->addEdge(13, 3, 8, 4, 5);
    type->addEdge(14, 4, 9, 5, 6);
  }

  // Octaedron12
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Octaedron12] = type;

    type->setInfos(this, IT_Octaedron12, "Octaedron12", 12, 18, 8);

    type->addFaceHexagon(0, 0, 5, 4, 3, 2, 1);
    type->addFaceHexagon(1, 6, 7, 8, 9, 10, 11);
    type->addFaceQuad(2, 0, 1, 7, 6);
    type->addFaceQuad(3, 1, 2, 8, 7);
    type->addFaceQuad(4, 2, 3, 9, 8);
    type->addFaceQuad(5, 3, 4, 10, 9);
    type->addFaceQuad(6, 4, 5, 11, 10);
    type->addFaceQuad(7, 5, 0, 6, 11);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 3, 4, 0);
    type->addEdge(3, 3, 4, 5, 0);
    type->addEdge(4, 4, 5, 6, 0);
    type->addEdge(5, 5, 0, 7, 0);
    type->addEdge(6, 6, 7, 1, 2);
    type->addEdge(7, 7, 8, 1, 3);
    type->addEdge(8, 8, 9, 1, 4);
    type->addEdge(9, 9, 10, 1, 5);
    type->addEdge(10, 10, 11, 1, 6);
    type->addEdge(11, 11, 6, 1, 7);
    type->addEdge(12, 0, 6, 7, 2);
    type->addEdge(13, 1, 7, 2, 3);
    type->addEdge(14, 2, 8, 3, 4);
    type->addEdge(15, 3, 9, 4, 5);
    type->addEdge(16, 4, 10, 5, 6);
    type->addEdge(17, 5, 11, 6, 7);
  }

  // HemiHexa7
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_HemiHexa7] = type;

    type->setInfos(this, IT_HemiHexa7, "HemiHexa7", 7, 11, 6);

    type->addFaceTriangle(0, 0, 1, 2);
    type->addFaceQuad(1, 0, 2, 3, 4);
    type->addFaceQuad(2, 0, 5, 6, 1);
    type->addFaceTriangle(3, 0, 4, 5);
    type->addFaceQuad(4, 1, 6, 3, 2);
    type->addFaceQuad(5, 3, 6, 5, 4);

    type->addEdge(0, 0, 1, 0, 2);
    type->addEdge(1, 1, 2, 0, 4);
    type->addEdge(2, 2, 0, 0, 1);
    type->addEdge(3, 2, 3, 1, 4);
    type->addEdge(4, 3, 4, 1, 5);
    type->addEdge(5, 4, 5, 3, 5);
    type->addEdge(6, 5, 0, 3, 2);
    type->addEdge(7, 0, 4, 3, 1);
    type->addEdge(8, 5, 6, 2, 5);
    type->addEdge(9, 6, 1, 2, 4);
    type->addEdge(10, 3, 6, 5, 4);
  }

  // HemiHexa6
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_HemiHexa6] = type;

    type->setInfos(this, IT_HemiHexa6, "HemiHexa6", 6, 10, 6);

    type->addFaceTriangle(0, 0, 1, 2);
    type->addFaceQuad(1, 0, 2, 3, 4);
    type->addFaceQuad(2, 0, 5, 3, 1);
    type->addFaceTriangle(3, 0, 4, 5);
    type->addFaceTriangle(4, 1, 3, 2);
    type->addFaceTriangle(5, 3, 5, 4);

    type->addEdge(0, 0, 1, 0, 2);
    type->addEdge(1, 1, 2, 0, 4);
    type->addEdge(2, 2, 0, 0, 1);
    type->addEdge(3, 2, 3, 1, 4);
    type->addEdge(4, 3, 4, 1, 5);
    type->addEdge(5, 4, 5, 3, 5);
    type->addEdge(6, 5, 0, 3, 2);
    type->addEdge(7, 0, 4, 3, 1);
    type->addEdge(8, 5, 3, 2, 5);
    type->addEdge(9, 3, 1, 2, 4);
  }

  // HemiHexa5
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_HemiHexa5] = type;

    type->setInfos(this, IT_HemiHexa5, "HemiHexa5", 5, 7, 4);

    type->addFaceTriangle(0, 0, 1, 2);
    type->addFaceQuad(1, 0, 2, 3, 4);
    type->addFaceQuad(2, 0, 4, 3, 1);
    type->addFaceTriangle(3, 1, 3, 2);

    type->addEdge(0, 0, 1, 0, 2);
    type->addEdge(1, 1, 2, 0, 3);
    type->addEdge(2, 2, 0, 0, 1);
    type->addEdge(3, 2, 3, 1, 3);
    type->addEdge(4, 3, 1, 2, 3);
    type->addEdge(5, 3, 4, 1, 2);
    type->addEdge(6, 4, 0, 1, 2);
  }

  // AntiWedgeLeft6
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_AntiWedgeLeft6] = type;

    type->setInfos(this, IT_AntiWedgeLeft6, "AntiWedgeLeft6", 6, 10, 6);

    type->addFaceTriangle(0, 0, 2, 1);
    type->addFaceQuad(1, 0, 3, 5, 2);
    type->addFaceQuad(2, 0, 1, 4, 3);
    type->addFaceTriangle(3, 3, 4, 5);
    type->addFaceTriangle(4, 1, 2, 4);
    type->addFaceTriangle(5, 2, 5, 4);

    type->addEdge(0, 0, 1, 0, 2);
    type->addEdge(1, 1, 2, 4, 0);
    type->addEdge(2, 2, 0, 1, 0);
    type->addEdge(3, 0, 3, 1, 2);
    type->addEdge(4, 1, 4, 2, 4);
    type->addEdge(5, 2, 5, 5, 1);
    type->addEdge(6, 3, 4, 3, 2);
    type->addEdge(7, 4, 5, 3, 5);
    type->addEdge(8, 5, 3, 3, 1);
    type->addEdge(9, 2, 4, 4, 5);
  }

  // AntiWedgeRight6
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_AntiWedgeRight6] = type;

    type->setInfos(this, IT_AntiWedgeRight6, "AntiWedgeRight6", 6, 10, 6);

    type->addFaceTriangle(0, 0, 2, 1);
    type->addFaceQuad(1, 0, 3, 5, 2);
    type->addFaceQuad(2, 0, 1, 4, 3);
    type->addFaceTriangle(3, 3, 4, 5);
    type->addFaceTriangle(4, 1, 2, 5);
    type->addFaceTriangle(5, 1, 5, 4);

    type->addEdge(0, 0, 1, 0, 2);
    type->addEdge(1, 1, 2, 4, 0);
    type->addEdge(2, 2, 0, 1, 0);
    type->addEdge(3, 0, 3, 1, 2);
    type->addEdge(4, 1, 4, 2, 5);
    type->addEdge(5, 2, 5, 4, 1);
    type->addEdge(6, 3, 4, 3, 2);
    type->addEdge(7, 4, 5, 3, 5);
    type->addEdge(8, 5, 3, 3, 1);
    type->addEdge(9, 1, 5, 5, 4);
  }

  // DiTetra5
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_DiTetra5] = type;

    type->setInfos(this, IT_DiTetra5, "DiTetra5", 5, 9, 6);

    type->addFaceTriangle(0, 0, 1, 3);
    type->addFaceTriangle(1, 1, 2, 3);
    type->addFaceTriangle(2, 2, 0, 3);
    type->addFaceTriangle(3, 1, 0, 4);
    type->addFaceTriangle(4, 2, 1, 4);
    type->addFaceTriangle(5, 0, 2, 4);

    type->addEdge(0, 0, 1, 0, 3);
    type->addEdge(1, 1, 2, 1, 4);
    type->addEdge(2, 2, 0, 2, 5);
    type->addEdge(3, 0, 3, 2, 0);
    type->addEdge(4, 1, 3, 0, 1);
    type->addEdge(5, 2, 3, 1, 2);
    type->addEdge(6, 0, 4, 3, 5);
    type->addEdge(7, 1, 4, 4, 3);
    type->addEdge(8, 2, 4, 5, 4);
  }

  // DualNode
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_DualNode] = type;

    type->setInfos(this, IT_DualNode, "DualNode", 1, 0, 0);
  }
  // DualEdge
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_DualEdge] = type;

    type->setInfos(this, IT_DualEdge, "DualEdge", 1, 0, 0);
  }
  // DualFace
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_DualFace] = type;

    type->setInfos(this, IT_DualFace, "DualFace", 1, 0, 0);
  }
  // DualCell
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_DualCell] = type;

    type->setInfos(this, IT_DualCell, "DualCell", 1, 0, 0);
  }
  // DualParticle
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_DualParticle] = type;

    type->setInfos(this, IT_DualParticle, "DualParticle", 1, 0, 0);
  }
  // Link
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Link] = type;

    type->setInfos(this, IT_Link, "Link", 0, 0, 0);
  }

  // Enneedron14
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Enneedron14] = type;

    type->setInfos(this, IT_Enneedron14, "IT_Enneedron14", 14, 21, 9);

    type->addFaceHeptagon(0, 0, 6, 5, 4, 3, 2, 1);
    type->addFaceHeptagon(1, 7, 8, 9, 10, 11, 12, 13);
    type->addFaceQuad(2, 0, 1, 8, 7);
    type->addFaceQuad(3, 1, 2, 9, 8);
    type->addFaceQuad(4, 2, 3, 10, 9);
    type->addFaceQuad(5, 3, 4, 11, 10);
    type->addFaceQuad(6, 4, 5, 12, 11);
    type->addFaceQuad(7, 5, 6, 13, 12);
    type->addFaceQuad(8, 6, 0, 7, 13);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 3, 4, 0);
    type->addEdge(3, 3, 4, 5, 0);
    type->addEdge(4, 4, 5, 6, 0);
    type->addEdge(5, 5, 6, 7, 0);
    type->addEdge(6, 6, 0, 8, 0);
    type->addEdge(7, 7, 8, 1, 2);
    type->addEdge(8, 8, 9, 1, 3);
    type->addEdge(9, 9, 10, 1, 4);
    type->addEdge(10, 10, 11, 1, 5);
    type->addEdge(11, 11, 12, 1, 6);
    type->addEdge(12, 12, 13, 1, 7);
    type->addEdge(13, 13, 7, 1, 8);
    type->addEdge(14, 0, 7, 8, 2);
    type->addEdge(15, 1, 8, 1, 2);
    type->addEdge(16, 2, 9, 2, 3);
    type->addEdge(17, 3, 10, 3, 4);
    type->addEdge(18, 4, 11, 4, 5);
    type->addEdge(19, 5, 12, 5, 6);
    type->addEdge(20, 6, 13, 6, 7);
  }
  // Decaedron16
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Decaedron16] = type;

    type->setInfos(this, IT_Decaedron16, "IT_Decaedron16", 16, 24, 10);

    type->addFaceOctogon(0, 0, 7, 6, 5, 4, 3, 2, 1);
    type->addFaceOctogon(1, 8, 9, 10, 11, 12, 13, 14, 15);
    type->addFaceQuad(2, 0, 1, 9, 8);
    type->addFaceQuad(3, 1, 2, 10, 9);
    type->addFaceQuad(4, 2, 3, 11, 10);
    type->addFaceQuad(5, 3, 4, 12, 11);
    type->addFaceQuad(6, 4, 5, 13, 12);
    type->addFaceQuad(7, 5, 6, 14, 13);
    type->addFaceQuad(8, 6, 7, 15, 14);
    type->addFaceQuad(9, 7, 0, 8, 15);

    type->addEdge(0, 0, 1, 2, 0);
    type->addEdge(1, 1, 2, 3, 0);
    type->addEdge(2, 2, 3, 4, 0);
    type->addEdge(3, 3, 4, 5, 0);
    type->addEdge(4, 4, 5, 6, 0);
    type->addEdge(5, 5, 6, 7, 0);
    type->addEdge(6, 6, 7, 8, 0);
    type->addEdge(7, 7, 0, 9, 0);
    type->addEdge(8, 8, 9, 1, 2);
    type->addEdge(9, 9, 10, 1, 3);
    type->addEdge(10, 10, 11, 1, 4);
    type->addEdge(11, 11, 12, 1, 5);
    type->addEdge(12, 12, 13, 1, 6);
    type->addEdge(13, 13, 14, 1, 7);
    type->addEdge(14, 14, 15, 1, 8);
    type->addEdge(15, 15, 8, 1, 9);
    type->addEdge(16, 0, 8, 9, 2);
    type->addEdge(17, 1, 9, 2, 3);
    type->addEdge(18, 2, 10, 3, 4);
    type->addEdge(19, 3, 11, 4, 5);
    type->addEdge(20, 4, 12, 5, 6);
    type->addEdge(21, 5, 13, 6, 7);
    type->addEdge(22, 6, 14, 7, 8);
    type->addEdge(23, 7, 15, 8, 9);
  }

  // Heptagon7
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Heptagon7] = type;

    type->setInfos(this, IT_Heptagon7, "Heptagon7", 7, 7, 7);

    type->addFaceLine(0, 0, 1);
    type->addFaceLine(1, 1, 2);
    type->addFaceLine(2, 2, 3);
    type->addFaceLine(3, 3, 4);
    type->addFaceLine(4, 4, 5);
    type->addFaceLine(5, 5, 6);
    type->addFaceLine(6, 6, 0);

    type->addEdge(0, 0, 1, 6, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 4, 2, 4);
    type->addEdge(4, 4, 5, 3, 5);
    type->addEdge(5, 5, 6, 4, 6);
    type->addEdge(6, 6, 0, 5, 0);
  }

  // Octogon8
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Octogon8] = type;

    type->setInfos(this, IT_Octogon8, "Octogon8", 8, 8, 8);

    type->addFaceLine(0, 0, 1);
    type->addFaceLine(1, 1, 2);
    type->addFaceLine(2, 2, 3);
    type->addFaceLine(3, 3, 4);
    type->addFaceLine(4, 4, 5);
    type->addFaceLine(5, 5, 6);
    type->addFaceLine(6, 6, 7);
    type->addFaceLine(7, 7, 0);

    type->addEdge(0, 0, 1, 7, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 4, 2, 4);
    type->addEdge(4, 4, 5, 3, 5);
    type->addEdge(5, 5, 6, 4, 6);
    type->addEdge(6, 6, 7, 5, 7);
    type->addEdge(7, 7, 0, 6, 0);
  }

  // Cell3D_Triangle3
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Cell3D_Triangle3] = type;

    type->setInfos(this, ItemTypeId(IT_Cell3D_Triangle3), "Cell3D_Triangle3", 2, 3, 3, 0);

    type->addEdge(0, 0, 1, 1, 2);
    type->addEdge(1, 1, 2, 2, 0);
    type->addEdge(2, 2, 0, 0, 1);
  }

  // Cell3D_Quad4
  {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    m_types[IT_Cell3D_Quad4] = type;

    type->setInfos(this, ItemTypeId(IT_Cell3D_Quad4), "Cell3D_Quad4", 2, 4, 4, 0);

    type->addEdge(0, 0, 1, 3, 1);
    type->addEdge(1, 1, 2, 0, 2);
    type->addEdge(2, 2, 3, 1, 3);
    type->addEdge(3, 3, 0, 2, 0);
  }

  { // Polygon & Polyhedron: generic item types
    String arcane_item_type_file = platform::getEnvironmentVariable("ARCANE_ITEM_TYPE_FILE");
    if (!arcane_item_type_file.null()) {
      // verify the existence of item type file. if doesn't exist return an exception
      readTypes(parallel_mng, arcane_item_type_file);
    }
  }

  // Calcul les relations face->arêtes
  // Cette opération doit être appelé en fin phase build
  for (Integer i = 0; i < m_types.size(); ++i) {
    ItemTypeInfoBuilder* type = static_cast<ItemTypeInfoBuilder*>(m_types[i]);
    if (!type)
      ARCANE_FATAL("ItemType '{0}' is not defined", type);
    type->computeFaceEdgeInfos();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeMng::
printTypes(std::ostream& ostr)
{
  ARCANE_ASSERT((m_initialized), ("Cannot use not built ItemTypeMng"));
  Integer nb_type = m_types.size();
  ostr << "** Number of types " << nb_type << '\n';
  for (Integer i = 0; i < nb_type; ++i) {
    ItemTypeInfo* type = m_types[i];
    ostr << " - Type " << type->typeId()
         << " Name: " << type->typeName()
         << " Nodes: " << type->nbLocalNode()
         << " Faces " << type->nbLocalFace() << '\n';
    for (Integer z = 0, sz = type->nbLocalFace(); z < sz; ++z) {
      ItemTypeInfo::LocalFace lf = type->localFace(z);
      ostr << " - - Face " << z << ":";
      for (Integer zk = 0, szk = lf.nbNode(); zk < szk; ++zk) {
        ostr << " " << lf.node(zk);
      }
      ostr << "\n";
    }
    ostr << "\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture d'un fichier de types voronoi.
 *
 *  Une cellule voronoi est un polytope dont le nombre de faces et de noeuds
 *  varie d'une cellule a l'autre. Ici, le type de chaque cellule est lu dans
 *  un fichier de types associe a un maillage d'entree donne.
 *  Ce fichier est passe dans la variable d'environnement: ARCANE_ITEM_TYPE_FILE
 *  Ex: setenv ARCANE_ITEM_TYPE_FILE PATH_TO_FILE/item_file
 *  Le format du fichier est la suivante:
 *
 *  nb_type
 *  type_id nb_faces nb_edges
 *  nb_node_face0 node0_face0 ... nodeN_face1
 *  .                 .               .
 *  .                 .               .
 *  .                 .               .
 *  nb_node_faceN node0_faceN ... nodeN_faceN
 *
 *  node0_edge0 node1_edge1 lefFace_edge0 rightFace_edge0
 *  .
 *  .
 *  .
 *  node0_edgeN node1_edge1 lefFace_edgeN rightFace_edgeN
 */
void ItemTypeMng::
readTypes(IParallelSuperMng* pm, const String& filename)
{
  m_trace->info() << "Reading additional item types from file '" << filename << "'";

  UniqueArray<Byte> bytes;
  Integer size = 0;

  // Lecture parallèle
  if (pm->commRank() == 0) {
    long unsigned int file_length = platform::getFileLength(filename);
    if (file_length == 0)
      throw IOException(A_FUNCINFO, "ARCANE_ITEM_TYPE_FILE is an empty file");
    std::ifstream ifile;
    ifile.open(filename.localstr(), std::ios::binary);
    if (ifile.fail())
      throw IOException(A_FUNCINFO, "Cannot open ARCANE_ITEM_TYPE_FILE item type file");
    bytes.resize(arcaneCheckArraySize(file_length + 1));
    ifile.read((char*)bytes.data(), file_length);
    bytes[(Integer)file_length] = '\0';
    if (ifile.bad())
      throw IOException(A_FUNCINFO, "Cannot read ARCANE_ITEM_TYPE_FILE item type file");
    size = bytes.size();
  }
  {
    IntegerArrayView bs(1, &size);
    pm->broadcast(bs, 0);
  }
  bytes.resize(size);
  if (size != 0) {
    pm->broadcast(bytes, 0);
  }
  else { // CC: add ending '\0'
    bytes.resize(1);
    bytes[0] = '\0';
  }

  // Already built polygons (size => identifier)
  typedef std::map<Integer, Integer> PolygonMapper;
  PolygonMapper built_polygons;
  built_polygons[3] = IT_Triangle3;
  built_polygons[4] = IT_Quad4;
  built_polygons[5] = IT_Pentagon5;
  built_polygons[6] = IT_Hexagon6;
  typedef std::set<Integer> KnownTypes;
  KnownTypes known_types;
  for (Integer i_type = 0; i_type < ItemTypeMng::nbBuiltInItemType(); ++i_type)
    known_types.insert(i_type);

  if (ItemTypeMng::nbBuiltInItemType() != m_types.size())
    throw FatalErrorException(A_FUNCINFO, "Invalid initialization of built-in item types");

  // Analyse du fichier de types
  std::istringstream ifile((char*)bytes.unguardedBasePointer(), std::istringstream::in);
  Integer nb_type;
  ifile >> nb_type;

  m_types.resize(ItemTypeMng::nbBuiltInItemType() + nb_type);
  Integer typeId, nbN, nbE, nbF;
  for (Integer i = 0; i < nb_type; ++i) {
    ItemTypeInfoBuilder* type = m_types_buffer->allocOne();
    ifile >> typeId >> nbF >> nbE;
    if (typeId >= nb_type || typeId < 0)
      throw IOException(A_FUNCINFO, String::format("Polyhedron reader cannot allow typeId {0}", typeId));
    typeId += ItemTypeMng::nbBuiltInItemType(); // translation d'indexation
    if (known_types.find(typeId) != known_types.end())
      throw FatalErrorException(A_FUNCINFO, String::format("Already existing typeId {0}", typeId));
    known_types.insert(typeId);
    if (nbE == nbF) // 2d case nbN == nbE == nbF
    {
      nbN = nbE;
    }
    else
      nbN = nbE - nbF + 2; // Calcul du nb de noeuds nbN a partir de nbE et nbF (formule d'Euler)

    type->setInfos(this, typeId, String::format("Polyhedron{0}_{1}-{2}-{3}", typeId, nbN, nbE, nbF), nbN, nbE, nbF);
    m_trace->debug(Trace::High) << "Adding " << type->typeName() << " type #"
                                << typeId - ItemTypeMng::nbBuiltInItemType() << " with " << nbN << " nodes, "
                                << nbE << " edges, " << nbF << " faces.";
    m_types[typeId] = type;
    for (Integer iface = 0; iface < nbF; ++iface) {
      ifile >> nbN;
      UniqueArray<Integer> nodeFace(nbN);
      for (Integer inodeFace = 0; inodeFace < nbN; ++inodeFace) {
        ifile >> nodeFace[inodeFace];
      }
      PolygonMapper::const_iterator finder = built_polygons.find(nbN);
      Integer face_type;
      if (finder != built_polygons.end()) {
        face_type = finder->second;
        m_trace->debug(Trace::High) << "\tAdding already existing face type " << face_type
                                    << " for face " << iface << " with " << nbN << " nodes";
      }
      else {
        ItemTypeInfoBuilder* type2 = m_types_buffer->allocOne();

        face_type = m_types.size();
        m_types.add(type2);
        built_polygons[nbN] = face_type;

        type2->setInfos(this, face_type, String::format("Polygon{0}", nbN), nbN, nbN, nbN);
        for (Integer j = 0; j < nbN; ++j)
          type2->addFaceLine(j, j, (j + 1) % nbN);

        for (Integer j = 0; j < nbN; ++j)
          type2->addEdge(j, j, (j + 1) % nbN, (j - 1 + nbN) % nbN, (j + 1) % nbN);
        m_trace->debug(Trace::High) << "\tAdding new face type " << face_type
                                    << " for face " << iface << " with " << nbN << " nodes";
      }
      type->addFaceGeneric(iface, face_type, nodeFace);
    }
    Integer node0, node1, leftFace, rightFace;
    for (Integer iedge = 0; iedge < nbE; ++iedge) {
      ifile >> node0 >> node1 >> leftFace >> rightFace;
      type->addEdge(iedge, node0, node1, leftFace, rightFace);
    }
  }

  m_trace->debug(Trace::High) << "Total number of types : " << m_types.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ItemTypeMng::
nbBasicItemType()
{
  return _singleton()->types().size();
}

Integer ItemTypeMng::
nbBuiltInItemType()
{
  return m_nb_builtin_item_type;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! AMR
Int32 ItemTypeMng::
nbHChildrenByItemType(Integer type)
{
  switch (type) {
  case IT_Line2:
    return 2;
  case IT_Triangle3:
    return 4;
  case IT_Quad4:
    return 4;
  case IT_Pentagon5:
    return 4;
  case IT_Hexagon6:
    return 4;
  case IT_Tetraedron4:
    return 8;
  case IT_Pyramid5:
    return 0;
  case IT_Pentaedron6:
    return 8;
  case IT_Hexaedron8:
    return 8;
  case IT_Heptaedron10:
  case IT_Octaedron12:
  case IT_HemiHexa7:
  case IT_HemiHexa6:
  case IT_HemiHexa5:
  case IT_AntiWedgeLeft6:
  case IT_AntiWedgeRight6:
  case IT_DiTetra5:
    return 0;
  default:
    ARCANE_FATAL("Not supported Item Type '{0}'", type);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeMng* ItemTypeMng::singleton_instance = 0;

ItemTypeMng* ItemTypeMng::
_singleton()
{
  if (!singleton_instance)
    singleton_instance = new ItemTypeMng();
  return singleton_instance;
}

void ItemTypeMng::
_destroySingleton()
{
  //GG: Ca plante avec Windows. Regarder pourquoi.
#ifndef ARCANE_OS_WIN32
  delete singleton_instance;
#endif
  singleton_instance = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<ItemTypeInfo*> ItemTypeMng::
types() const
{
  ARCANE_ASSERT((m_initialized), ("Cannot use not built ItemTypeMng"));
  return m_types;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeInfo* ItemTypeMng::
typeFromId(Integer id) const
{
  ARCANE_ASSERT((m_initialized), ("Cannot use not built ItemTypeMng"));
  return m_types[id];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemTypeInfo* ItemTypeMng::
typeFromId(ItemTypeId id) const
{
  ARCANE_ASSERT((m_initialized), ("Cannot use not built ItemTypeMng"));
  return m_types[id.typeId()];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ItemTypeMng::
typeName(Integer id) const
{
  return typeFromId(id)->typeName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ItemTypeMng::
typeName(ItemTypeId id) const
{
  return typeFromId(id)->typeName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Recopie de la fonction obsolète Item::typeName().
// TODO: voir pourquoi il y a un test sur nBasicItemType().
String ItemTypeMng::
_legacyTypeName(Integer t)
{
  if (t >= 0 && t < nbBasicItemType())
    return _singleton()->typeFromId(t)->typeName();
  return "InvalidType";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemTypeMng::hasGeneralCells(IMesh* mesh) const
{
  auto has_general_cells = false;
  if (m_mesh_with_general_cells.find(mesh) != m_mesh_with_general_cells.end()) {
    has_general_cells = true;
  }
  return has_general_cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemTypeMng::setMeshWithGeneralCells(IMesh* mesh) noexcept
{
  ARCANE_ASSERT(mesh, ("Trying to indicate a null mesh contains general cells."));
  m_mesh_with_general_cells.insert(mesh);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
