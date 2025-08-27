// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemRefinementPattern.h                                     (C) 2000-2020 */
/*                                                                           */
/* Fonctions utilitaires pour AMR.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMREFINEMENTPATTERN_H
#define ARCANE_ITEMREFINEMENTPATTERN_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Barton & Nackman Trick
template <class TypeImp>
class RefinementPatternT
{
public:

	Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
		return asImp().refine_matrix(i,j,k);
	}
  Integer face_mapping (const Integer i,const Integer j) const
  {
    return asImp().face_mapping(i,j);
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    return asImp().face_mapping_topo(i,j);
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    return asImp().hChildrenTypeId(i);
  }
  Integer getNbHChildren () const {
    return asImp().getNbHChildren();
  }

 protected:
	//! Barton & Nackman Trick
	TypeImp& asImp()
  {
		return static_cast<TypeImp&> (*this);
	}
	const TypeImp& asImp() const
  {
    return static_cast<const TypeImp&> (*this);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <int type_id> class ItemRefinementPatternT;
typedef ItemRefinementPatternT<IT_Quad4> Quad4RefinementPattern4Quad;
typedef ItemRefinementPatternT<IT_Hexaedron8> HexRefinementPattern8Hex ;
typedef ItemRefinementPatternT<IT_Tetraedron4> TetraRefinementPattern2Hex_2Penta_2Py_2Tetra ;
typedef ItemRefinementPatternT<IT_Pentaedron6> PrismRefinementPattern4Hex_4Pr ;
typedef ItemRefinementPatternT<IT_Pyramid5> PyramidRefinementPattern4Hex_4Py ;
typedef ItemRefinementPatternT<IT_HemiHexa5> HemiHex5RefinementPattern2Hex_4Penta_2HHex5 ;
typedef ItemRefinementPatternT<IT_HemiHexa6> HemiHex6RefinementPattern4Hex_4HHex7 ;
typedef ItemRefinementPatternT<IT_HemiHexa7> HemiHex7RefinementPattern6Hex_2HHex7 ;
typedef ItemRefinementPatternT<IT_AntiWedgeLeft6> AntiWedgeLeft6RefinementPattern4Hex_4HHex7 ;
typedef ItemRefinementPatternT<IT_AntiWedgeRight6> AntiWedgeRight6RefinementPattern4Hex_4HHex7 ;
typedef ItemRefinementPatternT<IT_DiTetra5> DiTetra5RefinementPattern2Hex_6HHex7 ;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_Quad4>
: public RefinementPatternT< ItemRefinementPatternT< IT_Quad4 > >
{
 public:
  ItemRefinementPatternT() {}

	//! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
		return _refine_matrix[i][j][k];
	}
  Integer face_mapping (const Integer i,const Integer j) const
  {
		return _face_mapping[i][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
		return _face_mapping_topo[i][j];
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Quad4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 4;

	static const Real _refine_matrix[4][4][4];
  static const Integer _face_mapping[4][4];
  static const Integer _face_mapping_topo[4][4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_Hexaedron8>
: public RefinementPatternT< ItemRefinementPatternT< IT_Hexaedron8 > >
{
 public:
  ItemRefinementPatternT() {}

	//! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
		return _refine_matrix [i][j][k];
	}
  Integer face_mapping (const Integer i,const Integer j) const
  {
    return _face_mapping [i][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    return _face_mapping_topo [i][j];
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Hexaedron8;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 8;

	static const Real _refine_matrix[8][8][8];
  static const Integer _face_mapping[8][6];
  static const Integer _face_mapping_topo[8][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HexRefinementPattern27Hex
: public RefinementPatternT<HexRefinementPattern27Hex>
{
 public:
	HexRefinementPattern27Hex() {}

	//! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer,const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Hexaedron8;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 27;
	static const double _refine_matrix_1[27][8][8];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_Tetraedron4> :
public RefinementPatternT< ItemRefinementPatternT< IT_Tetraedron4 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 2) return _refine_matrix_1 [i][j][k];
    if (i < 4) return _refine_matrix_2 [i-2][j][k];
    if (i < 6) return _refine_matrix_3 [i-4][j][k];
    return _refine_matrix_4 [i-6][j][k];
  }
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 2) return _face_mapping_1 [i][j];
    if (i < 4) return _face_mapping_2 [i-2][j];
    if (i < 6) return _face_mapping_3 [i-4][j];
    return _face_mapping_4 [i-6][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 2) return _face_mapping_topo_1 [i][j];
    if (i < 4) return _face_mapping_topo_2 [i-2][j];
    if (i < 6) return _face_mapping_topo_3 [i-4][j];
    return _face_mapping_topo_4 [i-6][j];
  }
  Integer hChildrenTypeId (const Integer i) const {
    if (i < 2) return IT_Hexaedron8;
    if (i < 4) return IT_Pentaedron6;
    if (i < 6) return IT_Pyramid5;
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;

  static const double _refine_matrix_1[2][8][4];
  static const Integer _face_mapping_1[2][6];
  static const Integer _face_mapping_topo_1[2][6];

  static const double _refine_matrix_2[2][6][4];
  static const Integer _face_mapping_2[2][5];
  static const Integer _face_mapping_topo_2[2][5];

  static const double _refine_matrix_3[2][5][4];
  static const Integer _face_mapping_3[2][5];
  static const Integer _face_mapping_topo_3[2][5];

  static const double _refine_matrix_4[2][4][4];
  static const Integer _face_mapping_4[2][4];
  static const Integer _face_mapping_topo_4[2][4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT TetraRefinementPattern8T
: public RefinementPatternT<TetraRefinementPattern8T>
{
 public:
	TetraRefinementPattern8T() {}

	//! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    return _refine_matrix_1 [i][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
	static const double _refine_matrix_1[8][4][4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT TetraRefinementPattern32T
: public RefinementPatternT<TetraRefinementPattern32T>
{
 public:
	TetraRefinementPattern32T() {}

	//! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer,const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 32;
	static const double _refine_matrix_1[32][4][4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_Pentaedron6> :
public RefinementPatternT< ItemRefinementPatternT< IT_Pentaedron6 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i<4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i<4) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-4][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i<4) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-4][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i<4) return IT_Hexaedron8;
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;

  static const double _refine_matrix_1[4][8][6];
  static const Integer _face_mapping_1[4][6];
  static const Integer _face_mapping_topo_1[4][6];

  static const double _refine_matrix_2[4][6][6];
  static const Integer _face_mapping_2[4][5];
  static const Integer _face_mapping_topo_2[4][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT PrismRefinementPattern8Pr
: public RefinementPatternT<PrismRefinementPattern8Pr>
{
 public:
	PrismRefinementPattern8Pr() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    return _refine_matrix_1 [i][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
	static const double _refine_matrix_1[8][6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT PrismRefinementPattern27Pr
: public RefinementPatternT<PrismRefinementPattern27Pr>
{
 public:
	PrismRefinementPattern27Pr() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer,const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 27;

  static const double _refine_matrix_1[27][6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_Pyramid5> :
public RefinementPatternT< ItemRefinementPatternT< IT_Pyramid5 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const      {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer i,const Integer j) const      {
    if (i < 4) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-4][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const      {
    if (i < 4) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-4][j];
  }
  Integer hChildrenTypeId (const Integer i) const {
    if (i < 4) return IT_Hexaedron8;
    return IT_Pyramid5;
  }
  Integer getNbHChildren () const {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 8;

  static const double _refine_matrix_1[4][8][5];
  static const Integer _face_mapping_1[4][6];
  static const Integer _face_mapping_topo_1[4][6];

  static const double _refine_matrix_2[4][5][5];
  static const Integer _face_mapping_2[4][5];
  static const Integer _face_mapping_topo_2[4][5];

};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT PyramidRefinementPattern4Py8T
: public RefinementPatternT<PyramidRefinementPattern4Py8T>
{
 public:
	PyramidRefinementPattern4Py8T() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_Pyramid5;
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 12;

	static const double _refine_matrix_1[4][5][5];

	static const double _refine_matrix_2[8][4][5];

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT PyramidRefinementPattern6Py4T
: public RefinementPatternT<PyramidRefinementPattern6Py4T>
{
 public:
	PyramidRefinementPattern6Py4T() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 6) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-6][j][k];
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer hChildrenTypeId (const Integer i) const {
    if (i < 6) return IT_Pyramid5;
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 10;

	static const double _refine_matrix_1[6][5][5];

	static const double _refine_matrix_2[4][4][5];

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT PyramidRefinementPattern4Py
: public RefinementPatternT<PyramidRefinementPattern4Py>
{
 public:
	PyramidRefinementPattern4Py() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer,const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Pyramid5;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 4;
	static const double _refine_matrix_1[4][5][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT PyramidRefinementPattern19Py12T
: public RefinementPatternT<PyramidRefinementPattern19Py12T>
{
 public:
	PyramidRefinementPattern19Py12T() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 19) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-19][j][k];
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 19) return IT_Pyramid5;
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 31;
	static const double _refine_matrix_1[19][5][5];
	static const double _refine_matrix_2[12][4][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_HemiHexa5> :
public RefinementPatternT< ItemRefinementPatternT< IT_HemiHexa5 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 2) return _refine_matrix_1 [i][j][k];
    if (i < 6) return _refine_matrix_2 [i-2][j][k];
    return _refine_matrix_3 [i-6][j][k];
  }
  //! mapping des orientations des faces des mailles filles avec les faces de la la maille mère
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 2) return _face_mapping_1 [i][j];
    if (i < 6) return _face_mapping_2 [i-2][j];
    return _face_mapping_3 [i-6][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 2) return _face_mapping_topo_1 [i][j];
    if (i < 6) return _face_mapping_topo_2 [i-2][j];
    return _face_mapping_topo_3 [i-6][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 2) return IT_Hexaedron8;
    if (i < 6) return IT_Pentaedron6;
    return IT_HemiHexa5;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 8;

  static const double _refine_matrix_1[2][8][5];
  static const Integer _face_mapping_1[2][6];
  static const Integer _face_mapping_topo_1[2][6];

  static const double _refine_matrix_2[4][6][5];
  static const Integer _face_mapping_2[4][5];
  static const Integer _face_mapping_topo_2[4][5];

  static const double _refine_matrix_3[2][5][5];
  static const Integer _face_mapping_3[2][4];
  static const Integer _face_mapping_topo_3[2][4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HemiHex5RefinementPattern3HHex5_2Pr_1HHex7
: public RefinementPatternT<HemiHex5RefinementPattern3HHex5_2Pr_1HHex7>
{
 public:
	HemiHex5RefinementPattern3HHex5_2Pr_1HHex7() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 3)
      return _refine_matrix_1 [i][j][k];
    if (i < 5)
      return _refine_matrix_2 [i-3][j][k];
    return _refine_matrix_3 [i-5][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw NotImplementedException(A_FUNCINFO);
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 3)
      return IT_HemiHexa5;
    if (i < 5)
      return IT_Pentaedron6;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 6;

	static const double _refine_matrix_1[3][5][5];

	static const double _refine_matrix_2[2][6][5];

	static const double _refine_matrix_3[1][7][5];

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HemiHex5RefinementPattern7HHex5_4Pr_3HHex6_2HHex7_1Hex
: public RefinementPatternT<HemiHex5RefinementPattern7HHex5_4Pr_3HHex6_2HHex7_1Hex>
{
 public:
	HemiHex5RefinementPattern7HHex5_4Pr_3HHex6_2HHex7_1Hex() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 7) return _refine_matrix_1 [i][j][k];
    if (i < 11) return _refine_matrix_2 [i-7][j][k];
    if (i < 14) return _refine_matrix_3 [i-11][j][k];
    if (i < 16) return _refine_matrix_4 [i-14][j][k];
    return _refine_matrix_5 [i-16][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw NotImplementedException(A_FUNCINFO);
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 7) return IT_HemiHexa5;
    if (i < 11) return IT_Pentaedron6;
    if (i < 14) return IT_HemiHexa6;
    if (i < 16) return IT_HemiHexa7;
    return IT_Hexaedron8;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 17;
	static const double _refine_matrix_1[7][5][5];
	static const double _refine_matrix_2[4][6][5];
	static const double _refine_matrix_3[3][6][5];
	static const double _refine_matrix_4[2][7][5];
	static const double _refine_matrix_5[1][8][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_HemiHexa6> :
public RefinementPatternT< ItemRefinementPatternT< IT_HemiHexa6 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 4) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-4][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 4) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-4][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_Hexaedron8;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 8;

  static const double _refine_matrix_1[4][8][6];
  static const Integer _face_mapping_1[4][6];
  static const Integer _face_mapping_topo_1[4][6];

  static const double _refine_matrix_2[4][7][6];
  static const Integer _face_mapping_2[4][6];
  static const Integer _face_mapping_topo_2[4][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HemiHex6RefinementPattern4HHex5_4HHex7
: public RefinementPatternT<HemiHex6RefinementPattern4HHex5_4HHex7>
{
 public:
	HemiHex6RefinementPattern4HHex5_4HHex7() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_HemiHexa5;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
	static const double _refine_matrix_1[4][5][6];
	static const double _refine_matrix_2[4][7][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HemiHex6RefinementPattern6HHex6_12HHex5_6HHex7_1Hex
: public RefinementPatternT<HemiHex6RefinementPattern6HHex6_12HHex5_6HHex7_1Hex>
{
 public:
	HemiHex6RefinementPattern6HHex6_12HHex5_6HHex7_1Hex() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 6) return _refine_matrix_1 [i][j][k];
    if (i < 18) return _refine_matrix_2 [i-6][j][k];
    if (i < 24) return _refine_matrix_3 [i-18][j][k];
    return _refine_matrix_4 [i-24][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw NotImplementedException(A_FUNCINFO);
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 6) return IT_HemiHexa6;
    if (i < 18) return IT_HemiHexa5;
    if (i < 24) return IT_HemiHexa7;
    return IT_Hexaedron8;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 25;
	static const double _refine_matrix_1[6][6][6];
	static const double _refine_matrix_2[12][5][6];
	static const double _refine_matrix_3[6][7][6];
	static const double _refine_matrix_4[1][8][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_HemiHexa7> :
public RefinementPatternT< ItemRefinementPatternT< IT_HemiHexa7 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 6) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-6][j][k];
  }
  //! mapping des orientations des faces des mailles filles avec les faces de la la maille mère
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 6) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-6][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 6) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-6][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 6) return IT_Hexaedron8;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;

  static const double _refine_matrix_1[6][8][7];
  static const Integer _face_mapping_1[6][6];
  static const Integer _face_mapping_topo_1[6][6];

  static const double _refine_matrix_2[2][7][7];
  static const Integer _face_mapping_2[2][6];
  static const Integer _face_mapping_topo_2[2][6];

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HemiHex7RefinementPattern4HHex7_4Pr_2HHex5_1Hex
: public RefinementPatternT<HemiHex7RefinementPattern4HHex7_4Pr_2HHex5_1Hex>
{
 public:
	HemiHex7RefinementPattern4HHex7_4Pr_2HHex5_1Hex() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    if (i < 8) return _refine_matrix_2 [i-4][j][k];
    if (i < 10) return _refine_matrix_3 [i-8][j][k];
    return _refine_matrix_4 [i-10][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_HemiHexa7;
    if (i < 8) return IT_Pentaedron6;
    if (i < 10) return IT_HemiHexa5;
    return IT_Hexaedron8;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 11;
	static const double _refine_matrix_1[4][7][7];
	static const double _refine_matrix_2[4][6][7];
	static const double _refine_matrix_3[2][5][7];
	static const double _refine_matrix_4[1][8][7];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT HemiHex7RefinementPattern8HHex7_14Pr_9HHex5_7Hex
: public RefinementPatternT<HemiHex7RefinementPattern8HHex7_14Pr_9HHex5_7Hex>
{
 public:
	HemiHex7RefinementPattern8HHex7_14Pr_9HHex5_7Hex() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 8) return _refine_matrix_1 [i][j][k];
    if (i < 22) return _refine_matrix_2 [i-8][j][k];
    if (i < 31) return _refine_matrix_3 [i-22][j][k];
    return _refine_matrix_4 [i-31][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 8) return IT_HemiHexa7;
    if (i < 22) return IT_Pentaedron6;
    if (i < 31) return IT_HemiHexa5;
    return IT_Hexaedron8;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 38;
	static const double _refine_matrix_1[8][7][7];
	static const double _refine_matrix_2[14][6][7];
	static const double _refine_matrix_3[9][5][7];
	static const double _refine_matrix_4[7][8][7];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_AntiWedgeLeft6> :
public RefinementPatternT< ItemRefinementPatternT< IT_AntiWedgeLeft6 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 4) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-4][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 4) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-4][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_Hexaedron8;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
  static const double _refine_matrix_1[4][8][6];
  static const Integer _face_mapping_1[4][6];
  static const Integer _face_mapping_topo_1[4][6];
  static const double _refine_matrix_2[4][7][6];
  static const Integer _face_mapping_2[4][6];
  static const Integer _face_mapping_topo_2[4][6];

};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT AntiWedgeLeft6RefinementPattern4AWL6_4Pr
: public RefinementPatternT<AntiWedgeLeft6RefinementPattern4AWL6_4Pr>
{
 public:
	AntiWedgeLeft6RefinementPattern4AWL6_4Pr() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_AntiWedgeLeft6;
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
	static const double _refine_matrix_1[4][6][6];
	static const double _refine_matrix_2[4][6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT AntiWedgeLeft6RefinementPattern9AWL6_18Pr
: public RefinementPatternT<AntiWedgeLeft6RefinementPattern9AWL6_18Pr>
{
 public:
	AntiWedgeLeft6RefinementPattern9AWL6_18Pr() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 9) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-9][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 9) return IT_AntiWedgeLeft6;
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 27;
	static const double _refine_matrix_1[9][6][6];
	static const double _refine_matrix_2[18][6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_AntiWedgeRight6> :
public RefinementPatternT< ItemRefinementPatternT< IT_AntiWedgeRight6 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 4) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-4][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 4) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-4][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_Hexaedron8;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
  static const double _refine_matrix_1[4][8][6];
  static const Integer _face_mapping_1[4][6];
  static const Integer _face_mapping_topo_1[4][6];
  static const double _refine_matrix_2[4][7][6];
  static const Integer _face_mapping_2[4][6];
  static const Integer _face_mapping_topo_2[4][6];

};
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT AntiWedgeRight6RefinementPattern4AWR6_4Pr
: public RefinementPatternT<AntiWedgeRight6RefinementPattern4AWR6_4Pr>
{
 public:
	AntiWedgeRight6RefinementPattern4AWR6_4Pr() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 4) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-4][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 4) return IT_AntiWedgeRight6;
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
	static const double _refine_matrix_1[4][6][6];
	static const double _refine_matrix_2[4][6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT AntiWedgeRight6RefinementPattern9AWR6_18Pr
: public RefinementPatternT<AntiWedgeRight6RefinementPattern9AWR6_18Pr>
{
 public:
	AntiWedgeRight6RefinementPattern9AWR6_18Pr() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 9) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-9][j][k];
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 9) return IT_AntiWedgeRight6;
    return IT_Pentaedron6;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 27;
	static const double _refine_matrix_1[9][6][6];
	static const double _refine_matrix_2[18][6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCANE_CORE_EXPORT ItemRefinementPatternT<IT_DiTetra5> :
public RefinementPatternT< ItemRefinementPatternT< IT_DiTetra5 > >
{
 public:
  ItemRefinementPatternT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 2) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-2][j][k];
  }
  //! mapping des orientations des faces des mailles filles avec les faces de la la maille mère
  Integer face_mapping (const Integer i,const Integer j) const
  {
    if (i < 2) return _face_mapping_1 [i][j];
    return _face_mapping_2 [i-2][j];
  }
  Integer face_mapping_topo (const Integer i,const Integer j) const
  {
    if (i < 2) return _face_mapping_topo_1 [i][j];
    return _face_mapping_topo_2 [i-2][j];
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 2) return IT_Hexaedron8;
    return IT_HemiHexa7;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 8;
  static const double _refine_matrix_1[2][8][5];
  static const Integer _face_mapping_1[2][6];
  static const Integer _face_mapping_topo_1[2][6];
  static const double _refine_matrix_2[6][7][5];
  static const Integer _face_mapping_2[6][6];
  static const Integer _face_mapping_topo_2[6][6];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DiTetra5RefinementPattern3DT_4Py_2T
: public RefinementPatternT<DiTetra5RefinementPattern3DT_4Py_2T>
{
 public:
	DiTetra5RefinementPattern3DT_4Py_2T() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 3) return _refine_matrix_1 [i][j][k];
    if (i < 7) return _refine_matrix_2 [i-3][j][k];
    return _refine_matrix_3 [i-7][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 3) return IT_DiTetra5;
    if (i < 7) return IT_Pyramid5;
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }

 private:

  static const Integer m_nb_hChildren = 9;
	static const double _refine_matrix_1[3][5][5];
	static const double _refine_matrix_2[4][5][5];
	static const double _refine_matrix_3[2][4][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DiTetra5RefinementPattern7DT_2T
: public RefinementPatternT<DiTetra5RefinementPattern7DT_2T>
{
 public:
	DiTetra5RefinementPattern7DT_2T() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer i,const Integer j,const Integer k) const
  {
    if (i < 7) return _refine_matrix_1 [i][j][k];
    return _refine_matrix_2 [i-7][j][k];
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer i) const
  {
    if (i < 7)
      return IT_DiTetra5;
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 9;
	static const double _refine_matrix_1[7][5][5];
	static const double _refine_matrix_2[2][4][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DiTetra5RefinementPattern64T
: public RefinementPatternT<DiTetra5RefinementPattern64T>
{
 public:
	DiTetra5RefinementPattern64T() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer,const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw FatalErrorException(A_FUNCINFO,"not implemented");
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 64;
	static const double _refine_matrix_1[64][4][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT DiTetra5RefinementPattern32DT
: public RefinementPatternT<DiTetra5RefinementPattern32DT>
{
 public:
	DiTetra5RefinementPattern32DT() {}

  //! matrice de transformation des noeuds de la maille mère en noeuds des mailles filles
  Real refine_matrix (const Integer,const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer face_mapping (const Integer,const Integer) const
  {
    throw NotSupportedException(A_FUNCINFO);
  }
  Integer hChildrenTypeId (const Integer) const
  {
    return IT_Tetraedron4;
  }
  Integer getNbHChildren () const
  {
    return m_nb_hChildren;
  }
 private:
  static const Integer m_nb_hChildren = 32;
	static const double _refine_matrix_1[32][5][5];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ARCANE_ITEMREFINEMENTPATTERN_H */
