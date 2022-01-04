// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeMng.h                                              (C) 2000-2014 */
/*                                                                           */
/* Classe gérant les GeomShape d'un maillage.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMSHAPEMNG_H
#define ARCANE_GEOMETRIC_GEOMSHAPEMNG_H
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
 * \brief Classe gérant les GeomShape des mailles d'un maillage.
 *
 * Cette classe stocke les informations des formes géométriques (GeomShape) associées
 * aux mailles du maillage. Pour une maille, la récupération d'une vue se fait
 * via la méthode initShape():
 \code
 GeomShapeMng shape_mng;
 Cell cell;
 GeomShapeView shape_view;
 // Initialise la vue \a shape_view sur la maille \a cell
 shape_mng.initShape(shape_view,cell);
 \endcode
 *
 * Une vue peut-être utilisée plusieurs fois. Par exemple, si on souhaite
 * itérer sur plusieurs mailles:
 \code
 * GeomShapeMng shape_mng;
 * GeomShapeView shape_view;
 * ENUMERATE_CELL(icell,allCells()){
 *   Cell cell = *icell;
 *   // Initialise la vue \a shape_view sur la maille \a cell
 *   shape_mng.initShape(shape_view,cell);
 *   info() << "Node0=" << shape_view.node(0);
 * }
 \endcode
 
 * La vue récupérée par GeomShapeView est constante. Pour récupérer une
 * vue modifiable, il faut utiliser mutableShapeView(). La vue modifiable
 * sert uniquement à mettre à jour les différentes
 * coordonnées (noeuds, centre des faces, ...).
 *
 * Avant de pouvoir utiliser une des méthodes initShape() ou mutableShapeView(),
 * il faut initialiser une des instance par l'appel à initialize().
 * L'initialisation effectue uniquement l'allocation mémoire mais ne met pas à jour
 * les coordonnées.
 * \warning La méthode initialize() doit aussi être appelée lorsque la topologie
 * du maillage change, par exemple après un ajout ou suppression de maille.
 *
 * Cette classe gère uniquement les données sur les formes géométriques et
 * ces dernières sont indépendantes des autres variables. Cela signifie
 * que si les coordonnées d'un noeud du maillage change, il faut explicitement
 * remettre à jour les informations de la forme géométrique. %Arcane fournit
 * la classe BarycentricGeomShapeComputer pour cela mais l'utilisateur
 * peut calculer ces informations d'une autre manière qu'en utilisant le barycentre. 
 *
 * Toutes les instances de cette classe dont le nom name() est identique
 * sont implicitement partagées et donc fournissent les mêmes GeomShapeView.
 * Par exemple:
 \code
 IMesh* mesh;
 GeomShapeMng shape_mng(mesh,"GenericElement");
 GeomShapeMng shape_mng2(shape_mng);
 // shape_mng et shape_mng2 partagent les mêmes GeomShapeView

 GeomShapeMng shape_mng3(mesh,"AleGenericElement");
 // shape_mng et shape_mng3 utilisent des valeurs différentes.
 \endcode
 *
 */
class ARCANE_CEA_GEOMETRIC_EXPORT GeomShapeMng
{
  // NOTE:
  // Comme cette classe peut-être utilisée par copie ou créée directement
  // via des noms de variable, elle ne doit pas contenir de champs
  // autres que des variables Arcane pour qu'il n'y ait pas d'incohérences
  // entre les différentes instances.

 public:
  
  //! Initialise pour le maillage \a mesh avec le nom \a name
  GeomShapeMng(IMesh* mesh,const String& name);
  //! Initialise pour le maillage \a mesh avec le nom par défaut \a GenericElement
  GeomShapeMng(IMesh* mesh);
  //! Constructeur de recopie.
  GeomShapeMng(const GeomShapeMng& rhs);

 public:

  //! Indique si l'instance est initialisée.
  bool isInitialized() const { return m_cell_shape_nodes.arraySize()!=0; }

  /*!
   * \brief Initialise l'instance.
   *
   * Il n'y a besoin d'initialiser qu'une seule fois les instances qui
   * ont le même nom.
   */
  void initialize();

  //! Initialise la vue \a ge avec les informations de la maille \a cell
  void initShape(GeomShapeView& ge,Cell cell) const
  {
    ge._setArray(m_cell_shape_nodes[cell].data(),m_cell_shape_faces[cell].data(),&m_cell_shape_centers[cell]);
    ge._setItem(cell);
  }

  //! Retourne une vue modifiable sur la GeomShape de la maille \a cell
  GeomShapeMutableView mutableShapeView(Cell cell)
  {
    return GeomShapeMutableView(m_cell_shape_nodes[cell].data(),m_cell_shape_faces[cell].data(),&m_cell_shape_centers[cell]);
  }

  //! Nom du gestionnaire.
  const String& name() const { return m_name; }

 private:

  String m_name;
  VariableCellArrayReal3 m_cell_shape_nodes; //!< Elements génériques noeuds
  VariableCellArrayReal3 m_cell_shape_faces; //!< Elements génériques face
  VariableCellReal3 m_cell_shape_centers; //!< Elements génériques centre
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef GeomShapeMng GeomCellMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
