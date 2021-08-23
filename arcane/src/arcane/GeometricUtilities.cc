// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricUtilities.cc                                       (C) 2000-2021 */
/*                                                                           */
/* Fonctions utilitaires sur la géométrie.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/MathUtils.h"
#include "arcane/GeometricUtilities.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real GeometricUtilities::QuadMapping::
computeInverseJacobian(Real3 uvw,Real3x3& matrix)
{
  Real3x3 jacobian = evaluateGradient(uvw);

  Real d3 = jacobian.x.x * jacobian.y.y - jacobian.x.y * jacobian.y.x;
  Real d2 = jacobian.x.z * jacobian.y.x - jacobian.x.y * jacobian.y.z;
  Real d1 = jacobian.x.y * jacobian.y.z - jacobian.x.z * jacobian.y.y;
	
  Real det = math::sqrt(d1*d1 + d2*d2 + d3*d3);
  if (math::isNearlyZero(det))
    return 0.;

  Real inv_det = 1 / det;

  jacobian.z.x = d1 * inv_det; 
  jacobian.z.y = d2 * inv_det; 
  jacobian.z.z = d3 * inv_det;

  matrix.x.x =  (jacobian.y.y * jacobian.z.z - jacobian.y.z * jacobian.z.y);
  matrix.y.x = -(jacobian.y.x * jacobian.z.z - jacobian.y.z * jacobian.z.x);
  matrix.z.x =  (jacobian.y.x * jacobian.z.y - jacobian.y.y * jacobian.z.x);
  
  matrix.x.y = -(jacobian.x.y * jacobian.z.z - jacobian.x.z * jacobian.z.y);
  matrix.y.y =  (jacobian.x.x * jacobian.z.z - jacobian.x.z * jacobian.z.x);
  matrix.z.y = -(jacobian.x.x * jacobian.z.y - jacobian.x.y * jacobian.z.x);
  
  matrix.x.z =  (jacobian.x.y * jacobian.y.z - jacobian.x.z * jacobian.y.y);
  matrix.y.z = -(jacobian.x.x * jacobian.y.z - jacobian.x.z * jacobian.y.x);
  matrix.z.z =  (jacobian.x.x * jacobian.y.y - jacobian.x.y * jacobian.y.x);

  matrix *= inv_det;

  return det;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Convertie une coordonnée cartérienne en coordonnée iso-paramétrique.
 *
 * Cette opération utilise un newton pour trouver la solution et peut donc
 * ne pas converger. Dans ce cas, elle retourne \a true.
 *
 * \param point position en coordonnée cartésienne du point à calculer.
 * \param uvw en retour, coordonnées iso-paramétriques calculées
 */

bool GeometricUtilities::QuadMapping::
cartesianToIso(Real3 point,Real3& uvw,ITraceMng* tm)
{
  const Real wanted_precision = m_precision;
  const Integer max_iteration = 1000;
	
  Real u_new, v_new, w_new;
  Real relative_error = 1.0 ;
  Integer nb_iter = 0;

  uvw = Real3(0.5,0.5,0.0);

  Real3x3 inverse_jacobian_matrix;

  while (relative_error>wanted_precision && nb_iter<max_iteration){
    ++nb_iter;
    
    computeInverseJacobian(uvw,inverse_jacobian_matrix);
    Real3 new_pos = evaluatePosition(uvw);
    u_new = uvw.x + inverse_jacobian_matrix.x.x * (point.x-new_pos.x)
      + inverse_jacobian_matrix.y.x * (point.y-new_pos.y)
      + inverse_jacobian_matrix.z.x * (point.z-new_pos.z);
    v_new = uvw.y + inverse_jacobian_matrix.x.y * (point.x-new_pos.x)
      + inverse_jacobian_matrix.y.y * (point.y-new_pos.y)
      + inverse_jacobian_matrix.z.y * (point.z-new_pos.z);
    //w_new = uvw.z + inverse_jacobian_matrix.x.z * (point.x-new_pos.x)
    //  + inverse_jacobian_matrix.y.z * (point.y-new_pos.y)
    //  + inverse_jacobian_matrix.z.z * (point.z-new_pos.z);
    w_new = 0.;
    
    relative_error = (u_new - uvw.x) * (u_new - uvw.x) + 
      (v_new - uvw.y) * (v_new - uvw.y) + 
      (w_new - uvw.z) * (w_new - uvw.z) ;

    if (tm)
      tm->info() << "CARTESIAN_TO_ISO I=" << nb_iter << " uvw=" << uvw
                 << " new_pos=" << new_pos
                 << " error=" << relative_error;

    uvw.x = u_new;
    uvw.y = v_new;
    uvw.z = w_new;
  }
  return (relative_error>wanted_precision);
}

Real3 GeometricUtilities::QuadMapping::
normal()
{
	Real3 normal= QuadMapping::_normal();
	Real norm = normal.x*normal.x + normal.y*normal.y + normal.z*normal.z;
	norm = math::sqrt(norm);
	normal /=norm;
	return normal;
}

static Real NormeLinf(Real3 v1,Real3 v2)
{
  Real3 v = v2 - v1;
  Real norm = math::max(math::abs(v.x),math::abs(v.y));
  norm = math::max(norm,math::abs(v.z));
  return norm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 GeometricUtilities::QuadMapping::
_normal()
{
  Real3 n0 = math::vecMul(m_pos[1] - m_pos[0],m_pos[2] - m_pos[0]);
  Real3 n1 = math::vecMul(m_pos[2] - m_pos[1],m_pos[3] - m_pos[1]);
  Real3 n2 = math::vecMul(m_pos[3] - m_pos[2],m_pos[0] - m_pos[2]);
  Real3 n3 = math::vecMul(m_pos[0] - m_pos[3],m_pos[1] - m_pos[3]);
  Real3 normal = (n0+n1+n2+n3) * 0.25;
  return normal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Convertie une coordonnée cartérienne en coordonnée iso-paramétrique.
 *
 * Cette opération utilise un newton pour trouver la solution et peut donc
 * ne pas converger. Dans ce cas, elle retourne \a true.
 *
 * \param point position en coordonnée cartésienne du point à calculer.
 * \param uvw en retour, coordonnées iso-paramétriques calculées
 */

bool GeometricUtilities::QuadMapping::
cartesianToIso2(Real3 point,Real3& uvw,ITraceMng* tm)
{
  Real epsi_newton = 1.0e-12;
  int newton_loop_limit = 100;
  bool error = false;

  // Position de départ pour le Newton
  Real u = 0.5;
  Real v = 0.5;
  Real w = 0.5;

  Real3 other_pos[4];
  Real3 normal = _normal();
  for( int i=0; i<4; ++i )
    other_pos[i] = m_pos[i] + normal;

  Real x0 = m_pos[0].x;
  Real y0 = m_pos[0].y;
  Real z0 = m_pos[0].z;
  Real x1 = m_pos[1].x;
  Real y1 = m_pos[1].y;
  Real z1 = m_pos[1].z;
  Real x2 = m_pos[2].x;
  Real y2 = m_pos[2].y;
  Real z2 = m_pos[2].z;
  Real x3 = m_pos[3].x;
  Real y3 = m_pos[3].y;
  Real z3 = m_pos[3].z;

  Real x4 = other_pos[0].x;
  Real y4 = other_pos[0].y;
  Real z4 = other_pos[0].z;
  Real x5 = other_pos[1].x;
  Real y5 = other_pos[1].y;
  Real z5 = other_pos[1].z;
  Real x6 = other_pos[2].x;
  Real y6 = other_pos[2].y;
  Real z6 = other_pos[2].z;
  Real x7 = other_pos[3].x;
  Real y7 = other_pos[3].y;
  Real z7 = other_pos[3].z;

  Real x = point.x;
  Real y = point.y;
  Real z = point.z;

  Real cx = 0.0;
  Real cy = 0.0;
  Real cz = 0.0;
  Real cxu = 0.0;
  Real cyu = 0.0;
  Real czu = 0.0;
  Real cxv = 0.0;
  Real cyv = 0.0;
  Real czv = 0.0;
  Real cxw = 0.0;
  Real cyw = 0.0;
  Real czw = 0.0;
  Real determinant = 0.0;
  
  // Boucle du NEWTON
  Real residue = 1.0e30;

  Real u0;
  Real v0;
  Real w0;

  int inewt=0;
  for(inewt=0; inewt<newton_loop_limit && residue>epsi_newton; inewt++ ) {

    u0 = u;
    v0 = v;
    w0 = w;

    cx =  + x0*((-1 + w0)*(-1 + u0)*(1 - v0)
                - w0*(-1 + u0)*(1 - v0)
                + (1 - w0)*u0*(1 - v0)
                + (-1 + w0)*(-1 + u0)*v0
                )
    + x1*(-(w0*(-1 + u0)*v0) + (1 - w0)*u0*v0)
    + x2*(w0*(-1 + u0)*v0 + w0*u0*v0)
    + x3*(-(w0*u0*(-1 + v0)) - w0*(-1 + u0)*v0)
    + x4*(-(w0*u0*(-1 + v0)) + (1 - w0)*u0*v0)
    + x5*((-1 + w0)*u0*v0 + w0*u0*v0)
    - x6*2*w0*u0*v0
    + x7*(-(w0*u0*(1 - v0)) + w0*u0*v0)
    ;

    cxv = x0*(1 - w0)*(-1 + u0)
    + x1*(-1 + w0)*(-1 + u0)
    - x2*w0*(-1 + u0)
    + x3*w0*(-1 + u0)
    + x4*(-1 + w0)*u0
    + x5*(1 - w0)*u0
    + x6*w0*u0
    - x7*w0*u0
    ;

    cxw =
    x0*(-1 + u0)*(1 - v0)
    + x1*(-1 + u0)*v0
    + x2*(1 - u0)*v0
    + x3*(-1 + u0)*(-1 + v0)
    + x4*u0*(-1 + v0)
    - x5*u0*v0
    + x6*u0*v0
    + x7*u0*(1 - v0)
    ;

    cxu =
    x0*(-1 + w0)*(1 - v0)
    + x1*(-1 + w0)*v0
    - x2*w0*v0
    + x3*w0*(-1 + v0)
    + x4*(-1 + w0)*(-1 + v0)
    + x5*(1 - w0)*v0
    + x6*w0*v0
    + x7*w0*(1 - v0)
    ;

    cy = 
    + y0*((-1 + w0)*(-1 + u0)*(1 - v0)
          - w0*(-1 + u0)*(1 - v0)
          + (1 - w0)*u0*(1 - v0)
          + (-1 + w0)*(-1 + u0)*v0
          )
    + y1*(-(w0*(-1 + u0)*v0) + (1 - w0)*u0*v0)
    + y2*(w0*(-1 + u0)*v0 + w0*u0*v0)
    + y3*(-(w0*u0*(-1 + v0)) - w0*(-1 + u0)*v0)
    + y4*(-(w0*u0*(-1 + v0)) + (1 - w0)*u0*v0)
    + y5*((-1 + w0)*u0*v0 + w0*u0*v0)
    - y6*2*w0*u0*v0
    + y7*(-(w0*u0*(1 - v0)) + w0*u0*v0)
    ;

    cyv =
    y0*(1 - w0)*(-1 + u0)
    + y1*(-1 + w0)*(-1 + u0)
    - y2*w0*(-1 + u0)
    + y3*w0*(-1 + u0)
    + y4*(-1 + w0)*u0
    + y5*(1 - w0)*u0
      + y6*w0*u0
    - y7*w0*u0
    ;

    cyw =
    y0*(-1 + u0)*(1 - v0)
    + y1*(-1 + u0)*v0
    + y2*(1 - u0)*v0
    + y3*(-1 + u0)*(-1 + v0)
    + y4*u0*(-1 + v0)
    - y5*u0*v0
    + y6*u0*v0
    + y7*u0*(1 - v0)
    ;

    cyu = y0*(-1 + w0)*(1 - v0)
    + y1*(-1 + w0)*v0
    - y2*w0*v0
    + y3*w0*(-1 + v0)
    + y4*(-1 + w0)*(-1 + v0)
    + y5*(1 - w0)*v0
    + y6*w0*v0
    + y7*w0*(1 - v0)
    ;

    cz = + z0*((-1 + w0)*(-1 + u0)*(1 - v0)
               - w0*(-1 + u0)*(1 - v0)
               + (1 - w0)*u0*(1 - v0)
               + (-1 + w0)*(-1 + u0)*v0
               )
    + z1*(-(w0*(-1 + u0)*v0) + (1 - w0)*u0*v0)
    + z2*(w0*(-1 + u0)*v0 + w0*u0*v0)
    + z3*(-(w0*u0*(-1 + v0)) - w0*(-1 + u0)*v0)
    + z4*(-(w0*u0*(-1 + v0)) + (1 - w0)*u0*v0)
    + z5*((-1 + w0)*u0*v0 + w0*u0*v0)
    - z6*2*w0*u0*v0
    + z7*(-(w0*u0*(1 - v0)) + w0*u0*v0)
    ;

    czv =
    z0*(1 - w0)*(-1 + u0)
    + z1*(-1 + w0)*(-1 + u0)
    - z2*w0*(-1 + u0)
    + z3*w0*(-1 + u0)
    + z4*(-1 + w0)*u0
    + z5*(1 - w0)*u0
    + z6*w0*u0
    - z7*w0*u0
    ;

    czw =
    z0*(-1 + u0)*(1 - v0)
    + z1*(-1 + u0)*v0
    + z2*(1 - u0)*v0
    + z3*(-1 + u0)*(-1 + v0)
    + z4*u0*(-1 + v0)
    - z5*u0*v0
    + z6*u0*v0
    + z7*u0*(1 - v0)
    ;

    czu =
    z0*(-1 + w0)*(1 - v0)
    + z1*(-1 + w0)*v0
    - z2*w0*v0
    + z3*w0*(-1 + v0)
    + z4*(-1 + w0)*(-1 + v0)
    + z5*(1 - w0)*v0
    + z6*w0*v0
    + z7*w0*(1 - v0)
    ;

    determinant =  cxv*cyu*czw - 
    cxu*cyv*czw - cxv*cyw*czu + 
    cxw*cyv*czu + cxu*cyw*czv - 
    cxw*cyu*czv;

    if (tm)
      tm->info() << "ITERATION DETERMINANT = " << determinant << " " << u
                 << " " << v << " " << w;

    if(math::abs(determinant)>1e-14){

      u  = (cxv*cyw*cz - cxw*cyv*cz - cxv*cy*czw
            + cx*cyv*czw + cxw*cy*czv - cx*cyw*czv
            + (- cyv*czw + cyw*czv) * x
            + (  cxv*czw - cxw*czv) * y
            + (- cxv*cyw + cxw*cyv) * z
            )/determinant ;
      
      v  = (-cxu*cyw*cz + cxw*cyu*cz + cxu*cy*czw
            - cx*cyu*czw - cxw*cy*czu + cx*cyw*czu
            + (  cyu*czw - cyw*czu) * x
            + (- cxu*czw + cxw*czu) * y
            + (  cxu*cyw - cxw*cyu) * z
            ) / determinant ;
      
      w  = -(cxv*cyu*cz - cxu*cyv*cz - cxv*cy*czu
             + cx*cyv*czu + cxu*cy*czv - cx*cyu*czv
             + (- cyv*czu + cyu*czv) * x
             + (  cxv*czu - cxu*czv) * y
             + (- cxv*cyu + cxu*cyv) * z
             )/determinant ;
      
      residue = NormeLinf(Real3(u,v,w),Real3(u0,v0,w0));
    }
    else {
      if (tm)
        tm->info() << "echec du Newton : déterminant nul";
      error = true;
      break;
    }
  } //for(int inewt=0; inewt<newton_loop_limit && residue>epsi_newton; inewt++ )

  if (inewt == newton_loop_limit) {
    if (tm)
      tm->info() << "Too many iterations in the newton";
    error = true;
  }
  // Cette routine calcule les coordonnées barycentriques entre 0 et 1 et
  // on veut la valeur entre -1 et 1
  Real3 iso(u,v,w);
  iso -= Real3(0.5,0.5,0.5);
  iso *= 2.0;
  uvw = Real3(iso.y,iso.z,0.0);
  return error;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeometricUtilities::ProjectionInfo GeometricUtilities::ProjectionInfo::
projection(Real3 v1,Real3 v2,Real3 v3,Real3 point)
{
  Real3 kDiff = v1 - point;
  Real3 edge0 = v2 - v1;
  Real3 edge1 = v3 - v1;
  Real fA00 = edge0.squareNormL2();
  Real fA01 = math::dot(edge0,edge1);
  Real fA11 = edge1.squareNormL2();
  Real fB0 = math::dot(kDiff,edge0);
  Real fB1 = math::dot(kDiff,edge1);
  Real fC = kDiff.squareNormL2();
  Real fDet = math::abs(fA00*fA11-fA01*fA01);
  Real alpha = fA01*fB1-fA11*fB0;
  Real beta = fA01*fB0-fA00*fB1;
  Real distance2 = 0.0;
  int region = -1;

  if ( alpha + beta <= fDet ){
    if ( alpha < 0.0 ){
      if ( beta < 0.0 ){ // region 4
        region = 4;
        if ( fB0 < 0.0 ){
          beta = 0.0;
          if ( -fB0 >= fA00 ){
            alpha = 1.0;
            distance2 = fA00+(2.0)*fB0+fC;
          }
          else{
            alpha = -fB0/fA00;
            distance2 = fB0*alpha+fC;
          }
        }
        else{
          alpha = 0.0;
          if ( fB1 >= 0.0 ){
            beta = 0.0;
            distance2 = fC;
          }
          else if ( -fB1 >= fA11 ){
            beta = 1.0;
            distance2 = fA11+(2.0)*fB1+fC;
          }
          else{
            beta = -fB1/fA11;
            distance2 = fB1*beta+fC;
          }
        }
      }
      else{  // region 3
        region = 3;
        alpha = 0.0;
        if ( fB1 >= 0.0 ){
          beta = 0.0;
          distance2 = fC;
        }
        else if ( -fB1 >= fA11 ){
          beta = 1.0;
          distance2 = fA11+(2.0)*fB1+fC;
        }
        else{
          beta = -fB1/fA11;
          distance2 = fB1*beta+fC;
        }
      }
    }
    else if ( beta < 0.0 ){  // region 5
      region = 5;
      beta = 0.0;
      if ( fB0 >= 0.0 ){
        alpha = 0.0;
        distance2 = fC;
      }
      else if ( -fB0 >= fA00 ){
        alpha = 1.0;
        distance2 = fA00+(2.0)*fB0+fC;
      }
      else{
        alpha = -fB0/fA00;
        distance2 = fB0*alpha+fC;
      }
    }
    else{  // region 0
      region = 0;
      // minimum at interior point
      Real fInvDet = (1.0)/fDet;
      alpha *= fInvDet;
      beta *= fInvDet;
      distance2 = alpha*(fA00*alpha+fA01*beta+(2.0)*fB0) +
        beta*(fA01*alpha+fA11*beta+(2.0)*fB1)+fC;
    }
  }
  else{
    if ( alpha < 0.0 ){  // region 2
      region = 2;
      Real fTmp0 = fA01 + fB0;
      Real fTmp1 = fA11 + fB1;
      if ( fTmp1 > fTmp0 ){
        Real fNumer = fTmp1 - fTmp0;
        Real fDenom = fA00-2.0f*fA01+fA11;
        if ( fNumer >= fDenom ){
          alpha = 1.0;
          beta = 0.0;
          distance2 = fA00+(2.0)*fB0+fC;
        }
        else{
          alpha = fNumer/fDenom;
          beta = 1.0 - alpha;
          distance2 = alpha*(fA00*alpha+fA01*beta+2.0f*fB0) +
            beta*(fA01*alpha+fA11*beta+(2.0)*fB1)+fC;
        }
      }
      else{
        alpha = 0.0;
        if (fTmp1 <= 0.0){
            beta = 1.0;
            distance2 = fA11+(2.0)*fB1+fC;
          }
        else if (fB1 >= 0.0){
          beta = 0.0;
          distance2 = fC;
        }
        else{
          beta = -fB1/fA11;
          distance2 = fB1*beta+fC;
        }
      }
    }
    else if (beta<0.0){  // region 6
      region = 6;
      Real fTmp0 = fA01 + fB1;
      Real fTmp1 = fA00 + fB0;
      if ( fTmp1 > fTmp0 ){
        Real fNumer = fTmp1 - fTmp0;
        Real fDenom = fA00-(2.0)*fA01+fA11;
        if (fNumer>=fDenom){
          beta = 1.0;
          alpha = 0.0;
          distance2 = fA11+(2.0)*fB1+fC;
        }
        else{
          beta = fNumer/fDenom;
          alpha = 1.0 - beta;
          distance2 = alpha*(fA00*alpha+fA01*beta+(2.0)*fB0) +
            beta*(fA01*alpha+fA11*beta+(2.0)*fB1)+fC;
        }
      }
      else{
        beta = 0.0;
        if (fTmp1 <= 0.0){
          alpha = 1.0;
          distance2 = fA00+(2.0)*fB0+fC;
        }
        else if (fB0>=0.0){
          alpha = 0.0;
          distance2 = fC;
        }
        else{
          alpha = -fB0/fA00;
          distance2 = fB0*alpha+fC;
        }
      }
    }
    else{  // region 1
      region = 1;
      Real fNumer = fA11 + fB1 - fA01 - fB0;
      if (fNumer<=0.0){
        alpha = 0.0;
        beta = 1.0;
        distance2 = fA11+(2.0)*fB1+fC;
      }
      else{
        Real fDenom = fA00-2.0f*fA01+fA11;
        if ( fNumer >= fDenom ){
          alpha = 1.0;
          beta = 0.0;
          distance2 = fA00+(2.0)*fB0+fC;
        }
        else{
          alpha = fNumer / fDenom;
          beta = 1.0 - alpha;
          distance2 = alpha*(fA00*alpha+fA01*beta+(2.0)*fB0) +
            beta*(fA01*alpha+fA11*beta+(2.0)*fB1)+fC;
        }
      }
    }
  }
  
  if (distance2<0.0)
    distance2 = 0.0;

  Real3 projection = v1 + alpha*edge0 + beta*edge1;
  
  return ProjectionInfo(distance2,region,alpha,beta,projection);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeometricUtilities::ProjectionInfo GeometricUtilities::ProjectionInfo::
projection(Real3 v1,Real3 v2,Real3 point)
{
  Real3 edge0 = v2 - v1;
  Real3 edge1 = point - v1;
  Real dot_product = math::dot(edge0,edge1);
  Real norm = edge0.squareNormL2();
  Real alpha = dot_product / norm;
  int region = -1;
  Real distance = 0.;
  if (alpha<0.){
    region = 1;
    distance = edge1.squareNormL2();
  }
  else if (alpha>1.){
    region = 2;
    distance = (point-v2).squareNormL2();
  }
  else{
    region = 0;
    distance = (point - (v1 + alpha*edge0)).squareNormL2();
  }
  Real3 projection = v1 + alpha*edge0;
#if 0
  cout << "InfosDistance: v1=" << v1 << " v2=" << v2 << " point="
       << point << " alpha=" << alpha
       << " distance=" << distance
       << " region=" << region
       << " projection=" << projection << '\n';
#endif
  return ProjectionInfo(distance,region,alpha,0.,projection);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GeometricUtilities::ProjectionInfo::
isInside(Real3 v1,Real3 v2,Real3 v3,Real3 point)
{
  Real3 kDiff = v1 - point;
  Real3 edge0 = v2 - v1;
  Real3 edge1 = v3 - v1;
  Real fA00 = edge0.squareNormL2();
  Real fA01 = math::dot(edge0,edge1);
  Real fA11 = edge1.squareNormL2();
  Real fB0 = math::dot(kDiff,edge0);
  Real fB1 = math::dot(kDiff,edge1);

  Real fDet = math::abs(fA00*fA11-fA01*fA01);

  Real alpha = fA01*fB1-fA11*fB0;
  Real beta = fA01*fB0-fA00*fB1;

  if ( alpha + beta <= fDet ){
    if ( alpha < 0.0 ){
      return false;
    }
    else if ( beta < 0.0 ){
      return false;
    }
    else{
      return true;
    }
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GeometricUtilities::ProjectionInfo::
isInside(Real3 v1,Real3 v2,Real3 point)
{
  Real3 edge0 = v2 - v1;
  Real3 edge1 = point - v1;
  Real dot_product = math::dot(edge0,edge1);
  Real norm = edge0.squareNormL2();
  //cout << "Infos: v1=" << v1 << " v2=" << v2 << " point="
  //<< point << " dot=" << dot_product << " norm2=" << norm << '\n';
  if (dot_product>0. && dot_product<norm)
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
