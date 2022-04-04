// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BarycentricGeomShapeComputer.h                              (C) 2000-2016 */
/*                                                                           */
/* Calcul des GeomShape en utilisant les barycentres.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_BARYCENTRICGEOMSHAPECOMPUTER_H
#define ARCANE_GEOMETRIC_BARYCENTRICGEOMSHAPECOMPUTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/VariableTypes.h"

#include "arcane/geometric/GeomShapeView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Calcul des GeomShape en utilisant les barycentres.
 *
 * Cette classe permet de mettre à jour les coordonnées des noeuds
 * d'un GeomShape et de calculer son centre et le centre de ses faces
 * en utilisant la formule du barycentre. Ces coordonnées doivent
 * être remises à jour dès qu'un des noeuds du maillage se déplace.
 *
 * Toutes les méthodes de cette classe sont statiques et il n'est donc
 * pas utile de créer des instances.
 *
 * Il existe plusieurs manières de mettre à jour:
 * - via computeAll(GeomShapeMng& shape_mng,VariableNodeReal3& coords,const CellGroup& cells),
 * auquel cas tous les GeomShape des mailles de \a cells sont mise à jour. C'est
 * la méthode la plus performante si on doit mettre à jour un grand nombre de
 * mailles.
 * - via computeAll(GeomShapeMutableView elem,const VariableNodeReal3& coords,Cell cell)
 * si on souhaite mettre à jour maille par maille.
 */
class ARCANE_CEA_GEOMETRIC_EXPORT BarycentricGeomShapeComputer
{
 public:

  //! Calcul les informations pour la maille \a cell
  static void computeAll(GeomShapeMutableView elem,const VariableNodeReal3& coords,Cell cell);

  //! Calcul les informations pour les mailles du groupe \a cells
  static void computeAll(GeomShapeMng& shape_mng,VariableNodeReal3& coords,const CellGroup& cells);

  /*!
   * \name Calcul du centre et des centres des faces par type de maille
   *
   * Les coordonnées des noeuds de \a elem doivent déjà avoir été positionnées.
   */
  ///@{
  static void computeTriangle3(GeomShapeMutableView elem);
  static void computeQuad4(GeomShapeMutableView elem);
  static void computeTetraedron4(GeomShapeMutableView elem);
  static void computePyramid5(GeomShapeMutableView elem);
  static void computePentaedron6(GeomShapeMutableView elem);
  static void computeHexaedron8(GeomShapeMutableView elem);
  static void computeHeptaedron10(GeomShapeMutableView elem);
  static void computeOctaedron12(GeomShapeMutableView elem);
  ///@}

  /*!
   * \brief Méthode template.
   *
   * Le paramètre template \a ItemType doit correspondre à un des types
   * suivants: GeomType::Triangle3, GeomType::Quad4, GeomType::Tetraedron4, GeomType::Pyramid5,
   * GeomType::Pentaedron6, GeomType::Hexaedron8, GeomType::Heptaedron10, GeomType::Octaedron12.
   *
   * Les coordonnées des noeuds de \a elem doivent déjà avoir été positionnées,
   * par exemple via l'appel à setNodes().
   *
   * L'appel se fait en spécifiant le type de maille tel que défini dans ArcaneTypes.h.
   * Par exemple, pour un Quad4:
   \code
   Cell cell = ...;
   GeomShapeMng shape_mng = ...;
   GeomShapeMutableView shape_view(shape_mng.mutableShapeView(cell));
   BarycentricGeomShapeComputer::compute<GeomType::Quad4>(shape_view);
   \endcode
   */
  template<GeomType ItemType> static
  void compute(GeomShapeMutableView elem);

  //! Remplit les informations des noeuds de la maille \a cell avec les coordonnées de \a node_coord.
  static void setNodes(GeomShapeMutableView elem,const VariableNodeReal3& node_coord,Cell cell)
  {
    Integer nb_node = cell.nbNode();
    for( Integer node_id=0; node_id<nb_node; ++node_id){
      elem.setNode(node_id,node_coord[cell.node(node_id)]);
    }
  }

 private:

  inline static void
  _setFace3D(Integer fid,GeomShapeMutableView& elem,Integer id1,Integer id2,Integer id3,Integer id4)
  {
    elem.setFace(fid, 0.25 * ( elem.node(id1) + elem.node(id2) + elem.node(id3) + elem.node(id4) ));
  }

  inline static void
  _setFace3D(Integer fid,GeomShapeMutableView& elem,Integer id1,Integer id2,Integer id3)
  {
    elem.setFace(fid, (1.0/3.0) * ( elem.node(id1) + elem.node(id2) + elem.node(id3) ));
  }

  inline static void
  _setFace2D(Integer fid,GeomShapeMutableView& elem,Integer id1,Integer id2)
  {
    elem.setFace(fid,
                 Real3( 0.5 * ( elem.node(id1).x + elem.node(id2).x ),
                        0.5 * ( elem.node(id1).y + elem.node(id2).y ),
                        0.0));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
