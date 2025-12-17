// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricUtilities.h                                        (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires sur la géométrie.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_GEOMETRICUTILITIES_H
#define ARCANE_CORE_GEOMETRICUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/utils/Real3x3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires sur la géométrie
 */
namespace Arcane::GeometricUtilities
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe permettant de mapper les coordonnées cartésiennes en
 * coordonnées barycentriques.
 *
 * Les coordonnées barycentriques sont comprises entre -1.0 et 1.0.
 *
 */
class ARCANE_CORE_EXPORT QuadMapping
{
 public:

  QuadMapping() = default;

 public:

  Real3 m_pos[4];
  Real m_precision = 1.0e-14;

 public:

  //! Calcule les coordonnées cartésiennes à partir des coordonnées iso-barycentriques
  Real3 evaluatePosition(Real3 iso) const
  {
    Real u = iso.x;
    Real v = iso.y;

    Real x0 = (1.0 - u) * (1.0 - v);
    Real x1 = (1.0 + u) * (1.0 - v);
    Real x2 = (1.0 + u) * (1.0 + v);
    Real x3 = (1.0 - u) * (1.0 + v);

    return 0.25 * (m_pos[0] * x0 + m_pos[1] * x1 + m_pos[2] * x2 + m_pos[3] * x3);
  }
  Real3x3 evaluateGradient(Real3 iso) const
  {
    Real u = iso.x;
    Real v = iso.y;

    Real t1 = 0.25 * (v - 1.0);
    Real t2 = 0.25 * (v + 1.0);
    Real t3 = 0.25 * (u - 1.0);
    Real t4 = 0.25 * (-u - 1.0);

    return Real3x3(m_pos[0] * t1 - m_pos[1] * t1 + m_pos[2] * t2 - m_pos[3] * t2,
                   m_pos[0] * t3 + m_pos[1] * t4 - m_pos[2] * t4 - m_pos[3] * t3,
                   Real3::null());
  }
  Real computeInverseJacobian(Real3 uvw, Real3x3& matrix);
  bool cartesianToIso(Real3 point, Real3& uvw, ITraceMng* tm);
  bool cartesianToIso2(Real3 point, Real3& uvw, ITraceMng* tm);
  Real3 normal();

 private:

  Real3 _normal();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Informations sur la projection d'un point sur un segment
 * ou un triangle.
 */
class ARCANE_CORE_EXPORT ProjectionInfo
{
 public:

  ProjectionInfo(Real distance, int region, Real alpha, Real beta, Real3 aprojection)
  : m_distance(distance)
  , m_region(region)
  , m_alpha(alpha)
  , m_beta(beta)
  , m_projection(aprojection)
  {}
  ProjectionInfo() = default;

 public:

  //! Distance du point à sa projection
  Real m_distance = FloatInfo<Real>::maxValue();
  //! Région dans laquelle se situe la projection (0 si intérieur au segment ou au triangle)
  int m_region = -1;
  //! Coordonnées barycentrique x de la projection
  Real m_alpha = -1.0;
  //! Coordonnées barycentrique y de la projection
  Real m_beta = -1.0;
  //! Position de la projection
  Real3 m_projection;

 public:

  //! Projection du point \a point au triangle défini  par \a v1, \a v2 et \a v3.
  static ProjectionInfo projection(Real3 v1, Real3 v2, Real3 v3, Real3 point);

  //! Projection du point \a point au segment défini  par \a v1, \a v2.
  static ProjectionInfo projection(Real3 v1, Real3 v2, Real3 point);

  /*! \brief Indique si un la projection du point \a point est à l'intérieur du triangle défini
     * par \a v1, \a v2 et \a v3.
     */
  static bool isInside(Real3 v1, Real3 v2, Real3 v3, Real3 point);
  /*! \brief Indique si un la projection du point \a point est à l'intérieur du segment défini
     * par \a v1 et \a v2.
     */
  static bool isInside(Real3 v1, Real3 v2, Real3 point);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::GeometricUtilities

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
