// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Connectivity.h                                              (C) 2000-2025 */
/*                                                                           */
/* Descripteur de connectivité.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CONNECTIVITY_H
#define ARCANE_CORE_CONNECTIVITY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/VariableTypes.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT Connectivity
{
 public:

  //! Flags de connectivité
  /*! 
   * Toutes les connectivités ne sont pas débrayables.
   * La numérotation jusqu'à CellToCell est faite pour la traduction avec _kindsToConnectivity 
   */
  enum eConnectivityType
  {
    CT_Null = 0,
    // From Node
    // CT_NodeToNode = 1<<1,
    CT_NodeToEdge = 1 << 2,
    CT_NodeToFace = 1 << 3, // Utilisé dans UnstructuredMeshUtilities::localIdsFromConnectivity
    CT_NodeToCell = 1 << 4,
    // From Edge
    CT_EdgeToNode = 1 << 5,
    // CT_EdgeToEdge = 1<<6,
    CT_EdgeToFace = 1 << 7,
    CT_EdgeToCell = 1 << 8,
    // From Face
    CT_FaceToNode = 1 << 9,
    CT_FaceToEdge = 1 << 10,
    CT_FaceToFace = 1 << 11,
    CT_FaceToCell = 1 << 12,
    // From Cell
    CT_CellToNode = 1 << 13,
    CT_CellToEdge = 1 << 14,
    CT_CellToFace = 1 << 15,
    // CT_CellToCell = 1<<16,
    CT_DoFToNode = 1 << 17,
    CT_DoFToEdge = 1 << 18,
    CT_DoFToFace = 1 << 19,
    CT_DoFToCell = 1 << 20,
    CT_DoFToDoF = 1 << 21,
    CT_DoFToParticle = 1 << 22,

    // Existing kind
    CT_HasNode = 1 << 23,
    CT_HasEdge = 1 << 24,
    CT_HasFace = 1 << 25,
    CT_HasCell = 1 << 26,

    // Frozen mark
    CT_Frozen = 1 << 27,

    // Dimension mark
    CT_Dim1D = 1 << 28,
    CT_Dim2D = 1 << 29,
    CT_Dim3D = 1 << 30,

    // Types composés
    // Connectivité par défaut
    CT_Default = CT_NodeToCell + CT_NodeToFace + CT_FaceToNode + CT_FaceToCell + CT_CellToNode + CT_CellToFace + CT_HasNode + CT_HasFace + CT_HasCell,

    CT_Default1D = CT_NodeToCell // minimum en 1D
    + CT_FaceToNode + CT_FaceToCell + CT_CellToNode + CT_CellToFace + CT_HasNode + CT_HasFace + CT_HasCell,

    CT_Default2D = CT_NodeToCell // minimum en 2D
    + CT_FaceToNode + CT_FaceToCell + CT_CellToNode + CT_CellToFace + CT_HasNode + CT_HasFace + CT_HasCell,

    CT_Default3D = CT_NodeToCell // minimum en 3D
    + CT_FaceToNode + CT_FaceToCell + CT_CellToNode + CT_CellToFace + CT_HasNode + CT_HasFace + CT_HasCell,

    CT_FullConnectivity2D = CT_NodeToFace + CT_NodeToCell + CT_FaceToNode + CT_FaceToCell + CT_CellToNode + CT_CellToFace + CT_HasNode + CT_HasFace + CT_HasCell,

    CT_FullConnectivity3D = CT_NodeToEdge + CT_NodeToFace + CT_NodeToCell + CT_EdgeToNode + CT_EdgeToFace + CT_EdgeToCell + CT_FaceToNode + CT_FaceToEdge + CT_FaceToCell + CT_FaceToFace + CT_CellToNode + CT_CellToEdge + CT_CellToFace +
    +CT_HasNode + CT_HasEdge + CT_HasFace + CT_HasCell,

    CT_EdgeConnectivity = CT_HasEdge + CT_NodeToEdge + CT_FaceToEdge + CT_CellToEdge +
    +CT_EdgeToNode + CT_EdgeToFace + CT_EdgeToCell,

    CT_GraphConnectivity = CT_DoFToNode + CT_DoFToEdge + CT_DoFToFace + CT_DoFToCell + CT_DoFToDoF + CT_DoFToParticle
  };

 public:

  //! Classe d'écriture d'un marqueur de connectivité
  class Printer
  {
   public:

    explicit Printer(const Integer connectivity)
    : m_connectivity(connectivity)
    {}

   public:

    void print(std::ostream& o) const;

   private:

    const Integer m_connectivity = 0;
  };

 public:

  /** Constructeur de la classe */
  explicit Connectivity(VariableScalarInteger connectivity);

 public:

  bool hasFace() const { return hasConnectivity(CT_HasFace); }
  bool hasEdge() const { return hasConnectivity(CT_HasEdge); }

  void enableConnectivity(Integer c);
  void disableConnectivity(Integer c);

  bool hasConnectivity(Integer c) const;
  static bool hasConnectivity(Integer connectivity, Integer c)
  {
    return _hasConnectivity(connectivity, c);
  }

  bool isFrozen() const;
  void freeze(IMesh* mesh);

  static Integer getPrealloc(Integer connectivity, eItemKind kindA, eItemKind kindB);

  //! Fonction d'écriture sur un flux
  static void print(std::ostream& o, Integer connectivity);

  //! Conversion de type en connectivité
  static Integer kindsToConnectivity(eItemKind kindA, eItemKind kindB);

 private:

  VariableScalarInteger m_connectivity;

 private:

  void _enableConnectivity(Integer c);
  void _disableConnectivity(Integer c);
  static inline bool _hasConnectivity(Integer connectivity, Integer c)
  {
    return ((connectivity & c) != 0);
  }

  static void _checkValid(Integer c);
  void _checkFrozen() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, const Connectivity::Printer& p);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_CONNECTIVITY_H */
