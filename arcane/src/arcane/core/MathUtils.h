// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathUtils.h                                                 (C) 2000-2024 */
/*                                                                           */
/* Fonctions mathématiques diverses.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATHUTILS_H
#define ARCANE_CORE_MATHUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Math.h"
#include "arcane/utils/ArrayView.h"
#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2.h"

#include "arcane/core/Algorithm.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Espace de nom pour les fonctions mathématiques.
 
  Cet espace de nom contient toutes les fonctions mathématiques utilisées
  par le code.
*/
namespace math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit vectoriel de \a u par \a v. dans \f$R^3\f$.
 *
 * \deprecated Utiliser cross() à la place.
 */
ARCCORE_HOST_DEVICE inline Real3
vecMul(Real3 u, Real3 v)
{
  return Real3(
	       u.y * v.z  - u.z * v.y,
	       u.z * v.x  - u.x * v.z,
	       u.x * v.y  - u.y * v.x
	       );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit vectoriel de \a u par \a v dans \f$R^2\f$.
 *
 * \deprecated Utiliser cross2D() à la place.
 */
ARCCORE_HOST_DEVICE inline Real
vecMul2D(Real3 u, Real3 v)
{
  return Real(u.x * v.y  - u.y * v.x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit vectoriel de \a u par \a v dans \f$R^2\f$.
 */
ARCCORE_HOST_DEVICE inline Real
cross2D(Real3 u,Real3 v)
{
  return Real(u.x * v.y  - u.y * v.x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit scalaire de \a u par \a v dans \f$R^2\f$.
 *
 * Il s'agit de: \f$u{\cdot}v\f$
 */
ARCCORE_HOST_DEVICE inline Real
dot(Real2 u,Real2 v)
{
  return (u.x * v.x  +  u.y * v.y );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit scalaire de \a u par \a v dans \f$R^2\f$
 *
 * \ingroup GroupMathUtils
 *
 * Il s'agit de: \f$u{\cdot}v\f$.
 *
 * \deprecated Utiliser dot(Real2,Real2) à la place
 */
ARCCORE_HOST_DEVICE inline Real
scaMul(Real2 u, Real2 v)
{
  return (u.x * v.x  +  u.y * v.y );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit scalaire de \a u par \a v
 *
 * \ingroup GroupMathUtils
 *
 * Il s'agit de: \f$u{\cdot}v\f$.
 */
ARCCORE_HOST_DEVICE inline Real
dot(Real3 u,Real3 v)
{
  return (u.x * v.x  +  u.y * v.y  +  u.z * v.z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit scalaire de \a u par \a v
 *
 * \ingroup GroupMathUtils
 *
 * Il s'agit de: \f$u{\cdot}v\f$
 *
 * \deprecated Utiliser dot(Real2,Real2) à la place
 */
ARCCORE_HOST_DEVICE inline Real
scaMul(Real3 u, Real3 v)
{
  return (u.x * v.x  +  u.y * v.y  +  u.z * v.z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit mixte de \a u, \a v et \a w
 *
 * \ingroup GroupMathUtils
 *
 */
ARCCORE_HOST_DEVICE inline Real
mixteMul(Real3 u,Real3 v,Real3 w)
{
  return dot(u,vecMul(v,w));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Déterminant de la matrice u,v,w
 */
ARCCORE_HOST_DEVICE inline Real
matDet(Real3 u,Real3 v,Real3 w)
{
  return (
	  (u.x * ( v.y*w.z - v.z*w.y )) +
	  (u.y * ( v.z*w.x - v.x*w.z )) +
	  (u.z * ( v.x*w.y - v.y*w.x ))
	  );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit tensoriel de deux vecteurs Real3.

 Il s'agit de \f$\vec{u}=~^t(u_{x},u_{y},u_{z})\f$ et \f$\vec{v}=~^t(v_{x},v_{y},v_{z})\f$
 et est noté \f$\vec{u} \otimes \vec{v}\f$, et est donné par~:

 *                Ux*Vx Ux*Vy Ux*Vz
 *U \otimes V =   Uy*Vx	Uy*Vy Uy*Vz
 *                Uz*Vx Uz*Vy Uz*Vz
 */
inline Real3x3
prodTens(Real3 u,Real3 v)
{
  return Real3x3(u.x*v,u.y*v,u.z*v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit matrice vecteur entre un tenseur et un vecteur.
 *
 * \ingroup GroupMathUtils
 *
 */
ARCCORE_HOST_DEVICE inline Real3
prodTensVec(Real3x3 t,Real3 v)
{
  return Real3(dot(t.x,v),dot(t.y,v),dot(t.z,v));
}
ARCCORE_HOST_DEVICE inline Real2
prodTensVec(Real2x2 t,Real2 v)
{
  return Real2(dot(t.x,v),dot(t.y,v));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit transposé(vecteur) matrice entre la transposée d'un vecteur et une matrice.
 *
 * Retourne le vecteur transposé du résultat.
 */
ARCCORE_HOST_DEVICE inline Real3
prodVecTens(Real3 v,Real3x3 t)
{
  return Real3(dot(v,Real3(t.x.x,t.y.x,t.z.x)),dot(v,Real3(t.x.y,t.y.y,t.z.y)),dot(v,Real3(t.x.z,t.y.z,t.z.z)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit matrice matrice entre deux tenseurs.
 */
ARCCORE_HOST_DEVICE inline Real3x3
matrixProduct(const Real3x3& t,const Real3x3& v)
{
  return Real3x3::fromLines(t.x.x*v.x.x+t.x.y*v.y.x+t.x.z*v.z.x,
                            t.x.x*v.x.y+t.x.y*v.y.y+t.x.z*v.z.y,
                            t.x.x*v.x.z+t.x.y*v.y.z+t.x.z*v.z.z,
                            t.y.x*v.x.x+t.y.y*v.y.x+t.y.z*v.z.x,
                            t.y.x*v.x.y+t.y.y*v.y.y+t.y.z*v.z.y,
                            t.y.x*v.x.z+t.y.y*v.y.z+t.y.z*v.z.z,
                            t.z.x*v.x.x+t.z.y*v.y.x+t.z.z*v.z.x,
                            t.z.x*v.x.y+t.z.y*v.y.y+t.z.z*v.z.y,
                            t.z.x*v.x.z+t.z.y*v.y.z+t.z.z*v.z.z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Transpose la matrice.
 * \deprecated Utiliser matrixTranspose() à la place.
 */
ARCCORE_HOST_DEVICE inline Real3x3
transpose(const Real3x3& t)
{
  return Real3x3(Real3(t.x.x, t.y.x, t.z.x),
                 Real3(t.x.y, t.y.y, t.z.y),
                 Real3(t.x.z, t.y.z, t.z.z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Transpose la matrice.
 */
ARCCORE_HOST_DEVICE inline Real3x3
matrixTranspose(const Real3x3& t)
{
  return Real3x3(Real3(t.x.x, t.y.x, t.z.x),
                 Real3(t.x.y, t.y.y, t.z.y),
                 Real3(t.x.z, t.y.z, t.z.z));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit doublement contracté de deux tenseurs.
 *
 * U:V = sum_{i,j \in \{x,y,z \}} U_{i,j}V_{i,j} 
 */
ARCCORE_HOST_DEVICE inline Real
doubleContraction(const Real3x3& u, const Real3x3& v)
{
  Real x1 = u.x.x * v.x.x;
  Real x2 = u.x.y * v.x.y;
  Real x3 = u.x.z * v.x.z;

  Real y1 = u.y.x * v.y.x;
  Real y2 = u.y.y * v.y.y;
  Real y3 = u.y.z * v.y.z;

  Real z1 = u.z.x * v.z.x;
  Real z2 = u.z.y * v.z.y;
  Real z3 = u.z.z * v.z.z;

  return x1 + x2 + x3 + y1 + y2 + y3 + z1 + z2 + z3;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Produit doublement contracté de deux tenseurs.
 *
 * U:V = sum_{i,j \in \{x,y,z \}} U_{i,j}V_{i,j} 
 */
ARCCORE_HOST_DEVICE inline Real
doubleContraction(const Real2x2& u,const Real2x2& v)
{
  Real x1 = u.x.x * v.x.x;
  Real x2 = u.x.y * v.x.y;

  Real y1 = u.y.x * v.y.x;
  Real y2 = u.y.y * v.y.y;

  return x1+x2+y1+y2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retourne le minimum de deux Real2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real2
min(Real2 a,Real2 b)
{
  return Real2( math::min(a.x,b.x), math::min(a.y,b.y) );
}
/*!
 * \brief Retourne le minimum de deux Real3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real3
min(Real3 a,Real3 b)
{
  return Real3( math::min(a.x,b.x), math::min(a.y,b.y), math::min(a.z,b.z) );
}
/*!
 * \brief Retourne le minimum de deux Real2x2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real2x2
min(const Real2x2& a,const Real2x2& b)
{
  return Real2x2( math::min(a.x,b.x), math::min(a.y,b.y) );
}
/*!
 * \brief Retourne le minimum de deux Real3x3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real3x3
min(const Real3x3& a,const Real3x3& b)
{
  return Real3x3( math::min(a.x,b.x), math::min(a.y,b.y), math::min(a.z,b.z) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le minimum de trois éléments.
 *
 * \ingroup GroupMathUtils
 *
 * Utilise l'opérateur < pour déterminer le minimum.
 */
template<class T> inline T
min(const T& a,const T& b,const T& c)
{
  return ( (a<b) ? ((a<c) ? a : ((b<c) ? b : c)) : ((b<c) ? b : c) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!\brief Retourne le maximum de trois éléments.
 *
 * \ingroup GroupMathUtils
 *
 * Utilise l'opérateur > pour déterminer le maximum.
 */
template<class T> inline T
max(const T& a,const T& b,const T& c)
{
  return ( (a>b) ? ((a>c) ? a : c) : ((b>c) ? b : c) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retourne le maximum de deux Real2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real2
max(Real2 a,Real2 b)
{
  return Real2( math::max(a.x,b.x), math::max(a.y,b.y) );
}
/*!
 * \brief Retourne le maximum de deux Real3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real3
max(Real3 a,Real3 b)
{
  return Real3( math::max(a.x,b.x), math::max(a.y,b.y), math::max(a.z,b.z) );
}
/*!
 * \brief Retourne le maximum de deux Real2x2
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real2x2
max(const Real2x2& a,const Real2x2& b)
{
  return Real2x2( math::max(a.x,b.x), math::max(a.y,b.y) );
}
/*!
 * \brief Retourne le maximum de deux Real3x3
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Real3x3
max(const Real3x3& a,const Real3x3& b)
{
  return Real3x3( math::max(a.x,b.x), math::max(a.y,b.y), math::max(a.z,b.z) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief retourne le min de quatre Real
 */
ARCCORE_HOST_DEVICE inline Real
min4Real(Real a,Real b,Real c,Real d)
{
  return min(min(a,b),min(c,d));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief retourne le max de quatre Real
 */
ARCCORE_HOST_DEVICE inline Real
max4Real(Real a,Real b,Real c,Real d)
{
  return max(max(a,b),max(c,d));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief retourne le min de huit <tt>Real</tt>
 */
ARCCORE_HOST_DEVICE inline Real
min8Real(const Real a[8])
{
  return min( min4Real(a[0],a[1],a[2],a[3]), min4Real(a[4],a[5],a[6],a[7]) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief retourne le max de huit <tt>Real</tt>
 */
inline Real max8Real(const Real a[8])
{
  return max( max4Real(a[0],a[1],a[2],a[3]), max4Real(a[4],a[5],a[6],a[7]) );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils 
 * \brief retourne le Min mod de quatre Real
 */
ARCCORE_HOST_DEVICE inline Real
minMod(Real a,Real b,Real c,Real d)
{
  Real zero = 0.;
  return min4Real(max(a,zero),max(b,zero),max(c,zero),max(d,zero))
    +max4Real(min(a,zero),min(b,zero),min(c,zero),min(d,zero));
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief retourne le Min mod de deux Reals
 */
ARCCORE_HOST_DEVICE inline Real
minMod2(Real a,Real b)
{
  Real zero = 0.;
  return min(max(a,zero),max(b,zero))+max(min(a,zero),min(b,zero));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief retourne le Max mod de deux Reals
 *
 */
ARCCORE_HOST_DEVICE inline Real
maxMod2(Real a,Real b)
{
  Real zero = 0.;
  return max(max(a,zero),max(b,zero))+min(min(a,zero),min(b,zero));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne l'erreur relative entre deux scalaires \a a et \a b.
 *
 * L'erreur relative est calculée par :
 *
 *	\f$\frac{a-b}{|a|+|b|}\f$
 */
inline Real
relativeError(Real a,Real b)
{
  Real sum = math::abs(a) + math::abs(b);
  return (isZero(sum)) ? (a-b) : (a-b)/sum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne l'erreur relative entre deux tenseurs \a T1 et \a T2.
 *
 * L'erreur relative est calculée comme le max des erreurs relatives
 * sur chacune des composantes des tenseurs.
 *
 */
inline Real
relativeError(const Real3x3& T1,const Real3x3& T2)
{
  Real err = 0;
  err = math::max(err,math::abs(relativeError(T1.x.x,T2.x.x)));
  err = math::max(err,math::abs(relativeError(T1.x.y,T2.x.y)));
  err = math::max(err,math::abs(relativeError(T1.x.z,T2.x.z)));
  err = math::max(err,math::abs(relativeError(T1.y.x,T2.y.x)));
  err = math::max(err,math::abs(relativeError(T1.y.y,T2.y.y)));
  err = math::max(err,math::abs(relativeError(T1.y.z,T2.y.z)));
  err = math::max(err,math::abs(relativeError(T1.z.x,T2.z.x)));
  err = math::max(err,math::abs(relativeError(T1.z.y,T2.z.y)));
  err = math::max(err,math::abs(relativeError(T1.z.z,T2.z.z)));

  return (err);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne l'erreur relative entre deux scalaires \a a et \a b relativement à \a b
 *
 * L'erreur relative est calculée par :
 *
 *  \f$\frac{a-b}{(|b|)}\f$
 */
inline Real
relativeError2(Real a, Real b)
{
  Real sum = math::abs(b);
  return (isZero(sum)) ? (a-b) : (a-b)/sum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne l'erreur relative entre deux scalaires \a a et \a b relativement à \a a
 *
 * L'erreur relative est calculée par :
 *
 *  \f$\frac{a-b}{(|b|)}\f$
 */
inline Real
relativeError1(Real a, Real b)
{
  Real sum = math::abs(a);
  return (isZero(sum)) ? (a-b) : (a-b)/sum;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche les valeurs extrèmes d'un tableau de couple (x,y).
 */
inline bool
searchExtrema(ConstArrayView<Real2> array,Real& xmin,Real& xmax,
              Real& ymin,Real& ymax,bool need_init)
{
  if (need_init){
    xmin = ymin = 0.;
    xmax = ymax = 1.;
  }

  Integer size = array.size();
  if (size==0)
    return false;

  if (need_init){
    xmin = xmax = array[0].x;
    ymin = ymax = array[0].y;
  }

  for( Integer i=1; i<size; ++i ){
    if (array[i].x < xmin)
      xmin = array[i].x;
    if (array[i].x > xmax)
      xmax = array[i].x;

    if (array[i].y < ymin)
      ymin = array[i].y;
    if (array[i].y > ymax)
      ymax = array[i].y;
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul du déterminant d'une matrice 3x3.
 */
ARCCORE_HOST_DEVICE inline Real
matrixDeterminant(Real3x3 m)
{
  return m.determinant();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Norme d'un vecteur
 *
 * \deprecated Utiliser Real3.abs() à la place.
 */
inline Real
normeR3(Real3 v1)
{
  Real norme = math::sqrt((v1.x)*(v1.x) + (v1.y)*(v1.y) + (v1.z)*(v1.z));
  return norme;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Matrice identite.
 */
ARCCORE_HOST_DEVICE inline Real3x3
matrix3x3Id()
{
  return Real3x3(Real3(1.0, 0.0, 0.0),
                 Real3(0.0, 1.0, 0.0),
                 Real3(0.0, 0.0, 1.0));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'inverse d'une matrice \a m en supposant connu son déterminant \a d
 */
ARCCORE_HOST_DEVICE inline Real3x3
inverseMatrix(const Real3x3& m,Real d)
{
  Real3x3 inv(Real3( m.y.y * m.z.z - m.y.z * m.z.y,   -m.x.y * m.z.z + m.x.z * m.z.y,    m.x.y * m.y.z - m.x.z * m.y.y),
              Real3( m.z.x * m.y.z - m.y.x * m.z.z,   -m.z.x * m.x.z + m.x.x * m.z.z,    m.y.x * m.x.z - m.x.x * m.y.z),
              Real3(-m.z.x * m.y.y + m.y.x * m.z.y,    m.z.x * m.x.y - m.x.x * m.z.y,   -m.y.x * m.x.y + m.x.x * m.y.y));
  inv /= d;
  return inv;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcul de l'inverse d'une matrice \a m.
 */
ARCCORE_HOST_DEVICE inline Real3x3
inverseMatrix(const Real3x3& m)
{
  Real d = m.determinant();
  return inverseMatrix(m,d);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit vectoriel de deux vecteurs à 3 composantes.
 *
 * \deprecated Utiliser cross() à la place.
 */
inline Real3
crossProduct3(Real3 v1, Real3 v2)
{
  Real3 v;
  v.x = v1.y*v2.z - v1.z*v2.y;
  v.y = v2.x*v1.z - v2.z*v1.x;
  v.z = v1.x*v2.y - v1.y*v2.x;
  
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils 
 * \brief Produit vectoriel de deux vecteurs à 3 composantes.
 */
ARCCORE_HOST_DEVICE inline Real3
cross(Real3 v1, Real3 v2)
{
  Real3 v;
  v.x = v1.y*v2.z - v1.z*v2.y;
  v.y = v2.x*v1.z - v2.z*v1.x;
  v.z = v1.x*v2.y - v1.y*v2.x;
  
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils 
 * \brief Normalisation d'un Real3.
 * \pre la norme de \a v ne doit pas être nulle.
 */
ARCCORE_HOST_DEVICE inline Real3
normalizeReal3(Real3 v)
{
  Real norme = math::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
      
  return Real3(v.x/norme, v.y/norme, v.z/norme);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils 
 * \brief Produit vectoriel normalisé.
 */
inline Real3
normalizedCrossProduct3(Real3 v1, Real3 v2)
{
  return normalizeReal3(cross(v1,v2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 *
 * \warning Cette méthode n'utilise pas la convention habituelle des Real3x3.
 * Elle suppose qu'ils sont rangés en colonne. En général, il faut utiliser
 * matrixTanspose() à la place.
 */
inline Real3x3
matrix3x3Transp(Real3x3 m)
{
  return Real3x3::fromColumns(m.x.x, m.x.y, m.x.z,
                              m.y.x, m.y.y, m.y.z,
                              m.z.x, m.z.y, m.z.z
                              );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Multiplication de 2 matrices 3x3.
 *
 * \warning Cette méthode n'utilise pas la convention habituelle des Real3x3.
 * Elle suppose qu'ils sont rangés en colonne. En général, il faut utiliser
 * matrixProduct() à la place.
 */
inline Real3x3 
matrix3x3Prod(Real3x3 m1, Real3x3 m2)
{
  return Real3x3::fromColumns( 
                              m1.x.x*m2.x.x + m1.y.x*m2.x.y + m1.z.x*m2.x.z,
                              m1.x.y*m2.x.x + m1.y.y*m2.x.y + m1.z.y*m2.x.z,
                              m1.x.z*m2.x.x + m1.y.z*m2.x.y + m1.z.z*m2.x.z,
                              m1.x.x*m2.y.x + m1.y.x*m2.y.y + m1.z.x*m2.y.z,
                              m1.x.y*m2.y.x + m1.y.y*m2.y.y + m1.z.y*m2.y.z,
                              m1.x.z*m2.y.x + m1.y.z*m2.y.y + m1.z.z*m2.y.z,
                              m1.x.x*m2.z.x + m1.y.x*m2.z.y + m1.z.x*m2.z.z,
                              m1.x.y*m2.z.x + m1.y.y*m2.z.y + m1.z.y*m2.z.z,
                              m1.x.z*m2.z.x + m1.y.z*m2.z.y + m1.z.z*m2.z.z 
                               );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Produit matrice 3x3 . vecteur 
 */
ARCCORE_HOST_DEVICE inline Real3
multiply(const Real3x3& m, Real3 v)
{
  return Real3( m.x.x*v.x + m.x.y*v.y + m.x.z*v.z,
                m.y.x*v.x + m.y.y*v.y + m.y.z*v.z,
                m.z.x*v.x + m.z.y*v.y + m.z.z*v.z 
                );
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie qu'une matrice vaut bien la matrice identité.
 */
inline bool
isNearlyId(Real3x3 m, Real epsilon = 1.e-10)
{
  static Real3x3 id = Real3x3(Real3(1., 0., 0.), Real3(0., 1., 0.), Real3(0., 0., 1.));
  //  static Real epsilon = 1.e-10;

  Real3x3 m0 = m - id;

  return (m0.x.x < epsilon) && (m0.x.y < epsilon) && (m0.x.z < epsilon) &&
  (m0.y.x < epsilon) && (m0.y.y < epsilon) && (m0.y.z < epsilon) &&
  (m0.z.x < epsilon) && (m0.z.y < epsilon) && (m0.z.z < epsilon);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Symétrie d'un vecteur \a u par rapport à un plan de normale \a n.
 */
inline Real3
planarSymmetric(Real3 u, Real3 n)
{
  Real3 u_tilde;
#ifdef ARCANE_CHECK
  if (n.normL2() == 0) {
    arcaneMathError(Convert::toDouble(n.normL2()), "planarSymetric");
  }
#endif
  Real3 norm = n / n.normL2();
  u_tilde = u - 2.0 * dot(norm, u) * norm;
  return u_tilde;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup GroupMathUtils
 * \brief Symétrie d'un vecteur u par rapport à un axe de vecteur directeur a .
 */ 
inline Real3
axisSymmetric(Real3 u,Real3 a)
{
  Real3 u_tilde;
#ifdef ARCANE_CHECK
  if (a.normL2()==0){
    arcaneMathError(Convert::toDouble(a.normL2()),"axisSymetric");
  }	
#endif
  Real3 norm = a / a.normL2();    
  u_tilde = 2.0 * dot(u,norm) * norm - u;
  return u_tilde;	
} 
    
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute le tableau \a copy_array dans l'instance.
 *
 * Comme aucune allocation mémoire n'est effectuée, le
 * nombre d'éléments de \a copy_array doit être inférieur ou égal au
 * nombre d'éléments courant. S'il est inférieur, les éléments du
 * tableau courant situés à la fin du tableau sont inchangés
 */
template<typename T> inline void
add(ArrayView<T> lhs,ConstArrayView<T> copy_array)
{
  Integer size = lhs.size();
  ARCANE_ASSERT( (copy_array.size()>=size), ("Bad size %d %d",copy_array.size(),size) );
  const T* copy_begin = copy_array.data();
  T* to_ptr = lhs.data();
  for( Integer i=0; i<size; ++i )
    to_ptr[i] += copy_begin[i];
}

/*!
 * \brief Ajoute le tableau \a copy_array dans l'instance.
 *
 * Comme aucune allocation mémoire n'est effectuée, le
 * nombre d'éléments de \a copy_array doit être inférieur ou égal au
 * nombre d'éléments courant. S'il est inférieur, les éléments du
 * tableau courant situés à la fin du tableau sont inchangés
 */
template<typename T> inline void
add(ArrayView<T> lhs,ArrayView<T> copy_array)
{
  Integer size = lhs.size();
  ARCANE_ASSERT( (copy_array.size()>=size), ("Bad size %d %d",copy_array.size(),size) );
  const T* copy_begin = copy_array.data();
  T* to_ptr = lhs.data();
  for( Integer i=0; i<size; ++i )
    to_ptr[i] += copy_begin[i];
}

/*!
 * \brief Soustrait le tableau \a copy_array de l'instance.
 *
 * Comme aucune allocation mémoire n'est effectuée, le
 * nombre d'éléments de \a copy_array doit être inférieur ou égal au
 * nombre d'éléments courant. S'il est inférieur, les éléments du
 * tableau courant situés à la fin du tableau sont inchangés
 */
template<typename T> inline void
sub(ArrayView<T> lhs,ConstArrayView<T> copy_array)
{
  Integer size = lhs.size();
  ARCANE_ASSERT( (copy_array.size()>=size), ("Bad size %d %d",copy_array.size(),size) );
  const T* copy_begin = copy_array.data();
  T* to_ptr = lhs.data();
  for( Integer i=0; i<size; ++i )
    to_ptr[i] -= copy_begin[i];
}

/*!
 * \brief Soustrait le tableau \a copy_array de l'instance.
 *
 * Comme aucune allocation mémoire n'est effectuée, le
 * nombre d'éléments de \a copy_array doit être inférieur ou égal au
 * nombre d'éléments courant. S'il est inférieur, les éléments du
 * tableau courant situés à la fin du tableau sont inchangés
 */
template<typename T> inline void
sub(ArrayView<T> lhs,ArrayView<T> copy_array)
{
  Integer size = lhs.size();
  ARCANE_ASSERT( (copy_array.size()>=size), ("Bad size %d %d",copy_array.size(),size) );
  const T* copy_begin = copy_array.data();
  T* to_ptr = lhs.data();
  for( Integer i=0; i<size; ++i )
    to_ptr[i] -= copy_begin[i];
}

/*!
 * \brief Multiplie terme à terme les éléments de l'instance par les
 * éléments du tableau \a copy_array.
 *
 * Comme aucune allocation
 * mémoire n'est effectuée, le nombre d'éléments de \a copy_array
 * doit être inférieur ou égal au nombre d'éléments courant. S'il
 * est inférieur, les éléments du tableau courant situés à la fin du
 * tableau sont inchangés
 */
template<typename T> inline void
mult(ArrayView<T> lhs,ConstArrayView<T> copy_array)
{
  Integer size = lhs.size();
  ARCANE_ASSERT( (copy_array.size()>=size), ("Bad size %d %d",copy_array.size(),size) );
  const T* copy_begin = copy_array.data();
  T* to_ptr = lhs.data();
  for( Integer i=0; i<size; ++i )
    to_ptr[i] *= copy_begin[i];
}

/*!
 * \brief Multiplie terme à terme les éléments de l'instance par les
 * éléments du tableau \a copy_array.
 *
 * Comme aucune allocation
 * mémoire n'est effectuée, le nombre d'éléments de \a copy_array
 * doit être inférieur ou égal au nombre d'éléments courant. S'il
 * est inférieur, les éléments du tableau courant situés à la fin du
 * tableau sont inchangés
 */
template<typename T> inline void
mult(ArrayView<T> lhs,ArrayView<T> copy_array)
{
  math::mult(lhs,copy_array.constView());
}

/*!
 * \brief Multiplie tous les éléments du tableau par le réel \a o.
 */
template<typename T> inline void
mult(ArrayView<T> lhs,T o)
{
  T* ptr = lhs.data();
  for( Integer i=0, size=lhs.size(); i<size; ++i )
    ptr[i] *= o;
}

/*!
 * \brief Met à la puissance \a o tous les éléments du tableau.
 */
template<typename T> inline void
power(ArrayView<T> lhs,T o)
{
  T* ptr = lhs.data();
  for( Integer i=0, size=lhs.size(); i<size; ++i )
    ptr[i] = math::pow(ptr[i],o);
}

} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
