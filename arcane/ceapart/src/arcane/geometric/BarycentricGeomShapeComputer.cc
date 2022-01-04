// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BarycentricGeomShapeComputer.cc                             (C) 2000-2014 */
/*                                                                           */
/* Calcul des GeomShape en utilisant les barycentres.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IMesh.h"

#include "arcane/AbstractItemOperationByBasicType.h"

#include "arcane/geometric/BarycentricGeomShapeComputer.h"
#include "arcane/geometric/GeomShapeMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille quadrangulaire.
 */
template <> void BarycentricGeomShapeComputer::
compute<GeomType::Quad4>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);

  // Calcule la position du centre.
  Real3 c = nul_vector;
      
  for( Integer i = 0; i<4; ++i ){
    c += elem.node(i);
  }
  elem.setCenter(0.25 * c);

  // Calcul la position des centres des faces.
  _setFace2D(0, elem, 0 , 1);
  _setFace2D(1, elem, 1 , 2);
  _setFace2D(2, elem, 2 , 3);
  _setFace2D(3, elem, 3 , 0);
}

/*!
 * \brief Calcul des positions des noeuds d'une maille quadrangulaire.
 */
void BarycentricGeomShapeComputer::
computeQuad4(GeomShapeMutableView elem)
{
  compute<GeomType::Quad4>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille triangulaire.
 *
 * Elle est considérée comme un quadrangle dégénéré.
 */
template <> void BarycentricGeomShapeComputer::
compute<GeomType::Triangle3>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);

  Real3 c = nul_vector;      

  // Calcule la position du centre.
  for( Integer i = 0; i<3; ++i )
    c += elem.node(i);
  
  elem.setCenter(c / 3.0);

  // Calcul la position des barycentres des faces.
  _setFace2D(0, elem, 0 , 1);
  _setFace2D(1, elem, 1 , 2);
  _setFace2D(2, elem, 2 , 0);
  
  elem.setFace(3, elem.node(0));
}

/*!
 * \brief Calcul des positions des noeuds d'une maille triangulaire.
 *
 * Elle est considérée comme un quadrangle dégénéré.
 */
void BarycentricGeomShapeComputer::
computeTriangle3(GeomShapeMutableView elem)
{
  compute<GeomType::Triangle3>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille hexaédrique.
 */
template <> void BarycentricGeomShapeComputer::
compute<GeomType::Hexaedron8>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);

  // Calcule la position du centre.
  Real3 c = nul_vector;
      
  for( Integer i = 0; i<8; ++i )
    c += elem.node(i);
      
  elem.setCenter(0.125 * c);
      
  // Calcul la position des centres des faces.
  _setFace3D(0, elem, 0 , 3 , 2 , 1);
  _setFace3D(1, elem, 0 , 4 , 7 , 3);
  _setFace3D(2, elem, 0 , 1 , 5 , 4);
  _setFace3D(3, elem, 4 , 5 , 6 , 7);
  _setFace3D(4, elem, 1 , 2 , 6 , 5);
  _setFace3D(5, elem, 2 , 3 , 7 , 6);
}

/*!
 * \brief Calcul des positions des noeuds d'une maille hexaédrique.
 */
void BarycentricGeomShapeComputer::
computeHexaedron8(GeomShapeMutableView elem)
{
  compute<GeomType::Hexaedron8>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille pyramidale.
 */
template<> void BarycentricGeomShapeComputer::
compute<GeomType::Pyramid5>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);
      
  // Calcule la position du centre.
  Real3 c = nul_vector;

  for( Integer i = 0; i<5; ++i )
    c += elem.node(i);
  elem.setCenter(0.2 * c);
  
  // Calcul la position des barycentres des faces.
  _setFace3D(0, elem, 0, 3, 2, 1);
  _setFace3D(1, elem, 0, 4, 3);
  _setFace3D(2, elem, 0, 1, 4);
  _setFace3D(3, elem, 1, 2, 4);
  _setFace3D(4, elem, 2, 3, 4);
      
  // Pour compatibilite avec pyra_face_connectic
  elem.setFace(5, elem.node(4));
}

/*!
 * \brief Calcul des positions des noeuds d'une maille pyramidale.
 */
void BarycentricGeomShapeComputer::
computePyramid5(GeomShapeMutableView elem)
{
  compute<GeomType::Pyramid5>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille pentaédrique.
 */
template<> void BarycentricGeomShapeComputer::
compute<GeomType::Pentaedron6>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);
      
  // Calcule la position du centre.
  Real3 c = nul_vector;

  for( Integer i = 0; i<6; ++i )
    c += elem.node(i);
	
  elem.setCenter( (1./6.) * c );

  _setFace3D(0, elem, 0, 2, 1);
  _setFace3D(1, elem, 0, 3, 5, 2);
  _setFace3D(2, elem, 0, 1, 4, 3);
  _setFace3D(3, elem, 3, 4, 5);
  _setFace3D(4, elem, 1, 2, 5, 4);
}

/*!
 * \brief Calcul des positions des noeuds d'une maille pentaédrique.
 */
void BarycentricGeomShapeComputer::
computePentaedron6(GeomShapeMutableView elem)
{
  compute<GeomType::Pentaedron6>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille tétraédrique.
 */
template<> void BarycentricGeomShapeComputer::
compute<GeomType::Tetraedron4>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);
      
  // Calcule la position du centre.
  Real3 c = nul_vector;
      
  for( Integer i = 0; i<4; ++i ){
    c += elem.node(i);
  }
      
  elem.setCenter(0.25 * c);

  _setFace3D(0, elem, 0, 2, 1);
  _setFace3D(1, elem, 0, 3, 2);
  _setFace3D(2, elem, 0, 1, 3);
  _setFace3D(3, elem, 3, 1, 2);
}

/*!
 * \brief Calcul des positions des noeuds d'une maille tétraédrique.
 */
void BarycentricGeomShapeComputer::
computeTetraedron4(GeomShapeMutableView elem)
{
  compute<GeomType::Tetraedron4>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille prismatique à base pentagonale.
 */
template<> void BarycentricGeomShapeComputer::
compute<GeomType::Heptaedron10>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);
      
  // Calcule la position du centre.
  Real3 c = nul_vector;
      
  for( Integer i = 0; i<10; ++i )
  {
    c += elem.node(i);
  }
  elem.setCenter( 0.1 * c );

  elem.setFace(0, 0.2  * ( elem.node(0) + elem.node(4) + elem.node(3) + elem.node(2) + elem.node(1) ));
  elem.setFace(1, 0.2  * ( elem.node(5) + elem.node(6) + elem.node(7) + elem.node(8) + elem.node(9) ));

  _setFace3D(2, elem, 0, 1, 6, 5);
  _setFace3D(3, elem, 1, 2, 7, 6);
  _setFace3D(4, elem, 2, 3, 8, 7);
  _setFace3D(5, elem, 3, 4, 9, 8);
  _setFace3D(6, elem, 4, 0, 5, 9);
}

/*!
 * \brief Calcul des positions des noeuds d'une maille prismatique à base pentagonale.
 */
void BarycentricGeomShapeComputer::
computeHeptaedron10(GeomShapeMutableView elem)
{
  compute<GeomType::Heptaedron10>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul des positions des noeuds d'une maille prismatique à base hexagonale.
 */
template<> void BarycentricGeomShapeComputer::
compute<GeomType::Octaedron12>(GeomShapeMutableView elem)
{
  const Real3 nul_vector = Real3(0.,0.,0.);
      
  // Calcule la position du centre.
  Real3 c = nul_vector;
      
  for( Integer i = 0; i<12; ++i ){
    c += elem.node(i);
  }
      
  elem.setCenter( (1./12.) * c );
      
  elem.setFace(0, (1./6.) * ( elem.node(0) + elem.node(5) + elem.node( 4) + elem.node( 3) + elem.node( 2) + elem.node( 1) ));
  elem.setFace(1, (1./6.) * ( elem.node(6) + elem.node(7) + elem.node( 8) + elem.node( 9) + elem.node(10) + elem.node(11) ));
  _setFace3D(2, elem, 0, 1,  7,  6);
  _setFace3D(3, elem, 1, 2,  8,  7);
  _setFace3D(4, elem, 2, 3,  9,  8);
  _setFace3D(5, elem, 3, 4, 10,  9);
  _setFace3D(6, elem, 4, 5, 11, 10);
  _setFace3D(7, elem, 5, 0,  6, 11);
}

/*!
 * \brief Calcul des positions des noeuds d'une maille prismatique à base hexagonale.
 */
void BarycentricGeomShapeComputer::
computeOctaedron12(GeomShapeMutableView elem)
{
  compute<GeomType::Octaedron12>(elem);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BarycentricGeomShapeComputer::
computeAll(GeomShapeMutableView elem,const VariableNodeReal3& coords,Cell cell)
{
  setNodes(elem,coords,cell);

  switch((GeomType)cell.type()){
  case GeomType::Hexaedron8: computeHexaedron8(elem); return;
  case GeomType::Pyramid5: computePyramid5(elem); return;
  case GeomType::Pentaedron6: computePentaedron6(elem); return;
  case GeomType::Tetraedron4: computeTetraedron4(elem); return;
  case GeomType::Heptaedron10: computeHeptaedron10(elem); return;
  case GeomType::Octaedron12: computeOctaedron12(elem); return;
  case GeomType::Quad4: computeQuad4(elem); return;
  case GeomType::Triangle3: computeTriangle3(elem); return;
  default :
    throw FatalErrorException(A_FUNCINFO,"Invalid cell type for compute");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BarycentricGeomShapeComputerByType
: public AbstractItemOperationByBasicType
{
  GeomShapeMng m_shape_mng;
  VariableNodeReal3 m_node_coords;
 public:
  BarycentricGeomShapeComputerByType(GeomShapeMng& shape_mng,VariableNodeReal3& node_coords)
  : m_shape_mng(shape_mng), m_node_coords(node_coords)
  {
  }

  template<GeomType ItemType> void
  _applyGeneric(ItemVectorView cells)
  {
    ENUMERATE_CELL(i_cell,cells){
      Cell cell = *i_cell;
      GeomShapeMutableView shape_view(m_shape_mng.mutableShapeView(cell));
      BarycentricGeomShapeComputer::setNodes(shape_view,m_node_coords,cell);
      BarycentricGeomShapeComputer::compute<ItemType>(shape_view);
    }
  }

  void applyVertex(ItemVectorView cells) override
  {
    ARCANE_UNUSED(cells);
    throw NotImplementedException(A_FUNCINFO);
  }
  void applyLine2(ItemVectorView cells) override
  {
    ARCANE_UNUSED(cells);
    throw NotImplementedException(A_FUNCINFO);
  }
  void applyPentagon5(ItemVectorView cells) override
  {
    ARCANE_UNUSED(cells);
    throw NotImplementedException(A_FUNCINFO);
  }
  void applyHexagon6(ItemVectorView cells) override
  {
    ARCANE_UNUSED(cells);
    throw NotImplementedException(A_FUNCINFO);
  }

  void applyQuad4(ItemVectorView cells) override
  {
    _applyGeneric<GeomType::Quad4>(cells);
  }
  void applyTriangle3(ItemVectorView cells) override
  {
    _applyGeneric<GeomType::Triangle3>(cells);
  }

  void applyHexaedron8(ItemVectorView cells) override
  {
    _applyGeneric<GeomType::Hexaedron8>(cells);
  }

  void applyPyramid5(ItemVectorView cells) override
  {
    _applyGeneric<GeomType::Pyramid5>(cells);
  }

  void applyPentaedron6(ItemVectorView cells) override
  { 
    _applyGeneric<GeomType::Pentaedron6>(cells);
  }

  void applyTetraedron4(ItemVectorView cells) override
  {   
    _applyGeneric<GeomType::Tetraedron4>(cells);
  }

  void applyHeptaedron10(ItemVectorView cells) override
  {
    _applyGeneric<GeomType::Heptaedron10>(cells);
  }

  void applyOctaedron12(ItemVectorView cells) override
  {
    _applyGeneric<GeomType::Octaedron12>(cells);
  }
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BarycentricGeomShapeComputer::
computeAll(GeomShapeMng& shape_mng,VariableNodeReal3& coords,const CellGroup& cells)
{
  BarycentricGeomShapeComputerByType s(shape_mng,coords);
  cells.applyOperation(&s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

