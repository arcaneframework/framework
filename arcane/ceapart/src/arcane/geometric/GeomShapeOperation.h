// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeOperation.h                                        (C) 2000-2014 */
/*                                                                           */
/* Opération sur une forme géométrique.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMETRICOPERATION_H
#define ARCANE_GEOMETRIC_GEOMETRICOPERATION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/AbstractItemOperationByBasicType.h"

#include "arcane/geometric/GeomShapeView.h"
#include "arcane/geometric/GeomShapeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneGeometric
 * \brief Classe template pour appliquer spécifique à une vue sur une forme géométrique.
 *
 * Cette classe permet de fournir un opérateur implémentant IItemOperationByBasicType
 * à partir d'une instance de OperationFunction qui utilise des vues
 * spécifiques sur des formes géométriques (les classes dérivées de GeomShapeView).
 *
 * La classe \a OperationFunction doit fournir une méthode apply() pour chaque type de forme
 * géométrique (Hexaedron8ShapeView, Quad4ShapeView, ...)
 *
 * L'appel se fait ensuite avec un groupe de mailles (CellGroup) en appelant
 * la méthode ItemGroup::applyOperation() avec cette instance en argument:
 *
 \code
 * // Définition de l'opération
 * class MyFunc
 * {
 *  public:
 *   void apply(Hexaedron8ShapeView view)
 *   {
 *     // Applique l'opération pour un hexaèdre.
 *   }
 * };
 *
 * GeomShapeOperation<MyFunc> op;
 * CellGroup cells;
 * // Applique \a op sur le groupe \a cells
 * cells.applyOperation(&op);
 \endcode
 */
template<typename OperationFunction>
class GeomShapeOperation
: public AbstractItemOperationByBasicType
{
 public:
  /*!
   * \brief Construit l'opérateur.
   *
   * Le premier argument est de type \a GeomShapeMng et sert à initialiser
   * l'opérateur. Les arguments suivants éventuels sont directement passés au 
   * constructeur de OperationFunction.
   *
   * \a shape_mng doit avoir été initialisé avant de pouvoir appliquer les opérations.
   */
  template<typename ... BuildArgs>
  GeomShapeOperation(GeomShapeMng& shape_mng,BuildArgs ... compute_function_args)
  : m_shape_mng(shape_mng),
    m_operation_function(compute_function_args ...)
  {
  }

  template<typename ShapeType>
  void apply(ItemVectorView cells)
  {
    ShapeType generic;
    ENUMERATE_CELL(i_cell,cells){
      Cell cell = *i_cell;
      m_shape_mng.initShape(generic,cell);
      m_operation_function.apply(generic);
    }
  }

  void applyTriangle3(ItemVectorView cells)
  {
    apply<TriangleShapeView>(cells);
  }
  void applyQuad4(ItemVectorView cells)
  {
    apply<QuadShapeView>(cells);
  }
  void applyPentagon5(ItemVectorView cells)
  {
    apply<PentagonShapeView>(cells);
  }
  void applyHexagon6(ItemVectorView cells)
  {
    apply<HexagonShapeView>(cells);
  }

  void applyTetraedron4(ItemVectorView cells)
  {
    apply<TetraShapeView>(cells);
  }
  void applyPyramid5(ItemVectorView cells)
  {
    apply<PyramidShapeView>(cells);
  }
  void applyPentaedron6(ItemVectorView cells)
  {
    apply<PentaShapeView>(cells);
  }
  void applyHexaedron8(ItemVectorView cells)
  {
    apply<HexaShapeView>(cells);
  }
  void applyHeptaedron10(ItemVectorView cells)
  {
    apply<Wedge7ShapeView>(cells);
  }
  void applyOctaedron12(ItemVectorView cells)
  {
    apply<Wedge8ShapeView>(cells);
  }

 public:
  //! Instance de l'opérateur
  OperationFunction& operation() { return m_operation_function; }
  //! Gestionnaire associé
  GeomShapeMng& cellShapeMng() { return m_shape_mng; }
 private:
  
  GeomShapeMng m_shape_mng;
  OperationFunction m_operation_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
