// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemTypeId.h                                                (C) 2000-2025 */
/*                                                                           */
/* Type d'une entité.                                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMTYPEID_H
#define ARCANE_CORE_ITEMTYPEID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Type d'une entité (Item).
 */
class ARCANE_CORE_EXPORT ItemTypeId
{
 public:

  ItemTypeId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemTypeId(Int16 id)
  : m_type_id(id)
  {}
  constexpr ARCCORE_HOST_DEVICE operator Int16() const { return m_type_id; }

 public:

  constexpr ARCCORE_HOST_DEVICE Int16 typeId() const { return m_type_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_type_id == IT_NullType; }
  /*!
   * \brief Créé une instance à partir d'un entier.
   *
   * Cette méthode lève une exception si \a v est supérieur à la valeur
   * maximale autorisée qui est actuellement 2^15.
   */
  static ItemTypeId fromInteger(Int64 v);

 private:

  Int16 m_type_id = IT_NullType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Numéro du type d'entité inconnu ou null
static constexpr ItemTypeId ITI_NullType(IT_NullType);
//! Numéro du type d'entité Noeud (1 sommet 1D, 2D et 3D)
static constexpr ItemTypeId ITI_Vertex(IT_Vertex);
//! Numéro du type d'entité Arête (2 sommets, 1D, 2D et 3D)
static constexpr ItemTypeId ITI_Line2(IT_Line2);
//! Numéro du type d'entité Triangle (3 sommets, 2D)
static constexpr ItemTypeId ITI_Triangle3(IT_Triangle3);
//! Numéro du type d'entité Quadrilatère (4 sommets, 2D)
static constexpr ItemTypeId ITI_Quad4(IT_Quad4);
//! Numéro du type d'entité Pentagone (5 sommets, 2D)
static constexpr ItemTypeId ITI_Pentagon5(IT_Pentagon5);
//! Numéro du type d'entité Hexagone (6 sommets, 2D)
static constexpr ItemTypeId ITI_Hexagon6(IT_Hexagon6);
//! Numéro du type d'entité Tetraèdre (4 sommets, 3D)
static constexpr ItemTypeId ITI_Tetraedron4(IT_Tetraedron4);
//! Numéro du type d'entité Pyramide (5 sommets, 3D)
static constexpr ItemTypeId ITI_Pyramid5(IT_Pyramid5);
//! Numéro du type d'entité Prisme (6 sommets, 3D)
static constexpr ItemTypeId ITI_Pentaedron6(IT_Pentaedron6);
//! Numéro du type d'entité Hexaèdre (8 sommets, 3D)
static constexpr ItemTypeId ITI_Hexaedron8(IT_Hexaedron8);
//! Numéro du type d'entité Heptaèdre (prisme à base pentagonale)
static constexpr ItemTypeId ITI_Heptaedron10(IT_Heptaedron10);
//! Numéro du type d'entité Octaèdre (prisme à base hexagonale)
static constexpr ItemTypeId ITI_Octaedron12(IT_Octaedron12);
//! Numéro du type d'entité HemiHexa7 (héxahèdre à 1 dégénérescence)
static constexpr ItemTypeId ITI_HemiHexa7(IT_HemiHexa7);
//! Numéro du type d'entité HemiHexa6 (héxahèdre à 2 dégénérescences non contigues)
static constexpr ItemTypeId ITI_HemiHexa6(IT_HemiHexa6);
//! Numéro du type d'entité HemiHexa5 (héxahèdres à 3 dégénérescences non contigues)
static constexpr ItemTypeId ITI_HemiHexa5(IT_HemiHexa5);
//! Numéro du type d'entité AntiWedgeLeft6 (héxahèdre à 2 dégénérescences contigues)
static constexpr ItemTypeId ITI_AntiWedgeLeft6(IT_AntiWedgeLeft6);
//! Numéro du type d'entité AntiWedgeRight6 (héxahèdre à 2 dégénérescences contigues (seconde forme))
static constexpr ItemTypeId ITI_AntiWedgeRight6(IT_AntiWedgeRight6);
//! Numéro du type d'entité DiTetra5 (héxahèdre à 3 dégénérescences orthogonales)
static constexpr ItemTypeId ITI_DiTetra5(IT_DiTetra5);
//! Numero du type d'entite noeud dual d'un sommet
static constexpr ItemTypeId ITI_DualNode(IT_DualNode);
//! Numero du type d'entite noeud dual d'une arête
static constexpr ItemTypeId ITI_DualEdge(IT_DualEdge);
//! Numero du type d'entite noeud dual d'une face
static constexpr ItemTypeId ITI_DualFace(IT_DualFace);
//! Numero du type d'entite noeud dual d'une cellule
static constexpr ItemTypeId ITI_DualCell(IT_DualCell);
//! Numéro du type d'entité liaison
static constexpr ItemTypeId ITI_Link(IT_Link);
//! Numéro du type d'entité Face pour les maillages 1D.
static constexpr ItemTypeId ITI_FaceVertex(IT_FaceVertex);
//! Numéro du type d'entité Cell pour les maillages 1D.
static constexpr ItemTypeId ITI_CellLine2(IT_CellLine2);
//! Numero du type d'entite noeud dual d'une particule
static constexpr ItemTypeId ITI_DualParticle(IT_DualParticle);

//! Numéro du type d'entité Enneèdre (prisme à base heptagonale)
static constexpr ItemTypeId ITI_Enneedron14(IT_Enneedron14);
//! Numéro du type d'entité Decaèdre (prisme à base Octogonale)
static constexpr ItemTypeId ITI_Decaedron16(IT_Decaedron16);

//! Numéro du type d'entité Heptagon 2D (heptagonale)
static constexpr ItemTypeId ITI_Heptagon7(IT_Heptagon7);

//! Numéro du type d'entité Octogon 2D (Octogonale)
static constexpr ItemTypeId ITI_Octogon8(IT_Octogon8);

//! Éléments quadratiques
//@{
//! Ligne d'ordre 2
static constexpr ItemTypeId ITI_Line3(IT_Line3);
//! Triangle d'ordre 2
static constexpr ItemTypeId ITI_Triangle6(IT_Triangle6);
//! Quadrangle d'ordre 2 (avec 4 noeuds sur les faces)
static constexpr ItemTypeId ITI_Quad8(IT_Quad8);
//! Tétraèdre d'ordre 2
static constexpr ItemTypeId ITI_Tetraedron10(IT_Tetraedron10);
//! Hexaèdre d'ordre 2
static constexpr ItemTypeId ITI_Hexaedron20(IT_Hexaedron20);
//! Hexaèdre d'ordre 2
static constexpr ItemTypeId ITI_Pentaedron15(IT_Pentaedron15);
//! Pyramide d'ordre 2
static constexpr ItemTypeId ITI_Pyramid13(IT_Pyramid13);
//@}

//! Maille Line3. EXPERIMENTAL !
static constexpr ItemTypeId ITI_CellLine3(IT_CellLine3);

/*!
 * \brief Mailles 2D dans un maillage 3D.
 * \warning Ces types sont expérimentaux et ne doivent
 * pas être utilisés en dehors de %Arcane.
 */
//@{
//! Maille Line2 dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Line2(IT_Cell3D_Line2);
//! Maille Triangulaire à 3 noeuds dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Triangle3(IT_Cell3D_Triangle3);
//! Maille Quadrangulaire à 4 noeuds dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Quad4(IT_Cell3D_Quad4);
//! Maille Line3 dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Line3(IT_Cell3D_Line3);
//! Maille Triangulaire à 6 noeuds dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Triangle6(IT_Cell3D_Triangle6);
//! Maille Quadrangulaire à 8 noeuds dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Quad8(IT_Cell3D_Quad8);
//! Maille Quadrangulaire à 9 noeuds dans un maillage 3D. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Cell3D_Quad9(IT_Cell3D_Quad9);
//@}

//! Quadrangle d'ordre 2 (avec 4 noeuds sur les faces et 1 noeud au centre). EXPERIMENTAL !
static constexpr ItemTypeId ITI_Quad9(IT_Quad9);
//! Hexaèdre d'ordre 2 (avec 12 noeuds sur les arêtes, 6 sur les faces et un noeud centre. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Hexaedron27(IT_Hexaedron27);

//! Ligne d'ordre 3. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Line4(IT_Line4);
//! Triangle d'ordre 3. EXPERIMENTAL !
static constexpr ItemTypeId ITI_Triangle10(IT_Triangle10);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
