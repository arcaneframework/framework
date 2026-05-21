// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HllcSchemeTypes.h                                           (C) 2000-2026 */
/*                                                                           */
/* Types et constantes pour le schéma HLLC.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANETEST_HLLCSCHEMETYPES_H
#define ARCANETEST_HLLCSCHEMETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Real3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest::HllcScheme
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Types pour le schéma hydro HLLC.
 */
class TypesHllcScheme
{
 public:

  //! Type de condition aux limites
  enum eBoundaryCondition
  {
    Wall,       //!< Paroi glissante (vitesse normale nulle)
    Inflow,     //!< Entrée (vitesse et densité imposées)
    Outflow,    //!< Sortie libre (gradient nul)
    Unknown     //!< Type inconnu
  };

  //! Type de limiteur pour la reconstruction MUSCL
  enum eLimiter
  {
    MinMod,
    VanLeer,
    VanAlbada,
    Superbee
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief État conservatif: densité, quantité de mouvement, énergie totale.
 */
struct ConservativeState
{
  Real density = 0.0;
  Real3 momentum = Real3::zero();
  Real energy = 0.0;

  ARCCORE_HOST_DEVICE Real internalEnergy() const
  {
    return energy - Real(0.5) * math::dot(momentum, momentum) / density;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief État primitif: densité, vitesse, pression.
 */
struct PrimitiveState
{
  Real density = 0.0;
  Real3 velocity = Real3::zero();
  Real pressure = 0.0;

  ARCCORE_HOST_DEVICE Real speedOfSound(Real gamma) const
  {
    return math::sqrt(gamma * pressure / density);
  }

  ARCCORE_HOST_DEVICE Real totalEnergy(Real gamma) const
  {
    Real ek = Real(0.5) * density * math::dot(velocity, velocity);
    return pressure / (gamma - Real(1.0)) + ek;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Flux à travers une face de normale donnée.
 */
struct FluxState
{
  Real density_flux = 0.0;
  Real3 momentum_flux = Real3::zero();
  Real energy_flux = 0.0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule l'état primitif à partir de l'état conservatif et de gamma.
 */
inline ARCCORE_HOST_DEVICE PrimitiveState
conservativeToPrimitive(const ConservativeState& c, Real gamma)
{
  PrimitiveState p;
  p.density = c.density;
  p.velocity = c.momentum / c.density;
  Real ek = Real(0.5) * c.density * math::dot(p.velocity, p.velocity);
  p.pressure = (gamma - Real(1.0)) * (c.energy - ek);
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule l'état conservatif à partir de l'état primitif.
 */
inline ARCCORE_HOST_DEVICE ConservativeState
primitiveToConservative(const PrimitiveState& p, Real gamma)
{
  ConservativeState c;
  c.density = p.density;
  c.momentum = p.density * p.velocity;
  Real ek = Real(0.5) * p.density * math::dot(p.velocity, p.velocity);
  c.energy = p.pressure / (gamma - Real(1.0)) + ek;
  return c;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule le flux d'Euler 3D à travers une face de normale \a n.
 *
 * Le flux retourné est projeté sur la normale: F·n.
 */
inline ARCCORE_HOST_DEVICE FluxState
eulerFlux(const PrimitiveState& p, const Real3& normal, Real gamma)
{
  Real vn = math::dot(p.velocity, normal);
  Real p_total = p.pressure;
  Real ek = Real(0.5) * p.density * math::dot(p.velocity, p.velocity);
  Real e_total = p_total / (gamma - Real(1.0)) + ek;

  FluxState f;
  f.density_flux = p.density * vn;
  f.momentum_flux = p.density * vn * p.velocity + p_total * normal;
  f.energy_flux = vn * (e_total + p_total);
  return f;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Solveur de Riemann HLLC.
 *
 * Calcule le flux numérique à l'interface entre les états gauche et droit
 * en utilisant le solveur HLL avec les vitesses d'onde améliorées
 * (estimateur d'Einfeldt via la moyenne de Roe).
 *
 * \param left  état primitif gauche
 * \param right état primitif droit
 * \param normal normale unitaire à la face (pointant de gauche vers droite)
 * \param gamma coefficient adiabatique
 * \return flux numérique projeté sur la normale
 */
inline ARCCORE_HOST_DEVICE FluxState
hllcFlux(const PrimitiveState& left, const PrimitiveState& right,
         const Real3& normal, Real gamma)
{
  // Vitesses normales
  Real ul = math::dot(left.velocity, normal);
  Real ur = math::dot(right.velocity, normal);

  // Vitesses du son
  Real cl = left.speedOfSound(gamma);
  Real cr = right.speedOfSound(gamma);

  // Enthalpies totales
  Real hl = (left.totalEnergy(gamma) + left.pressure) / left.density;
  Real hr = (right.totalEnergy(gamma) + right.pressure) / right.density;

  // Moyenne de Roe
  Real sqrt_rho_l = math::sqrt(left.density);
  Real sqrt_rho_r = math::sqrt(right.density);
  Real sum_sqrt_rho = sqrt_rho_l + sqrt_rho_r;

  Real u_tilde = (sqrt_rho_l * ul + sqrt_rho_r * ur) / sum_sqrt_rho;
  Real h_tilde = (sqrt_rho_l * hl + sqrt_rho_r * hr) / sum_sqrt_rho;
  Real c_tilde = math::sqrt((gamma - Real(1.0)) * (h_tilde - Real(0.5) * u_tilde * u_tilde));

  // Vitesses d'onde HLLC (Einfeldt)
  Real s_l = math::min(ul - cl, u_tilde - c_tilde);
  Real s_r = math::max(ur + cr, u_tilde + c_tilde);

  // États conservatifs gauche et droit
  ConservativeState wl = primitiveToConservative(left, gamma);
  ConservativeState wr = primitiveToConservative(right, gamma);

  // Flux d'Euler gauche et droit
  FluxState fl = eulerFlux(left, normal, gamma);
  FluxState fr = eulerFlux(right, normal, gamma);

  FluxState f_hll;

  if (s_l >= Real(0.0)) {
    f_hll = fl;
  }
  else if (s_r <= Real(0.0)) {
    f_hll = fr;
  }
  else {
    Real inv_ds = Real(1.0) / (s_r - s_l);

    f_hll.density_flux = (s_r * fl.density_flux - s_l * fr.density_flux
                          + s_l * s_r * (wr.density - wl.density)) * inv_ds;
    f_hll.momentum_flux = (s_r * fl.momentum_flux - s_l * fr.momentum_flux
                           + s_l * s_r * (wr.momentum - wl.momentum)) * inv_ds;
    f_hll.energy_flux = (s_r * fl.energy_flux - s_l * fr.energy_flux
                         + s_l * s_r * (wr.energy - wl.energy)) * inv_ds;
  }

  return f_hll;
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Limiteur MUSCL basé sur le rapport des pentes.
 */
inline ARCCORE_HOST_DEVICE Real
limiter(TypesHllcScheme::eLimiter type, Real r)
{
  switch (type) {
  case TypesHllcScheme::MinMod:
    return math::max(Real(0.0), math::min(Real(1.0), r));
  case TypesHllcScheme::VanLeer:
    if (r <= Real(0.0))
      return Real(0.0);
    return (Real(2.0) * r) / (Real(1.0) + r);
  case TypesHllcScheme::VanAlbada:
    if (r <= Real(0.0))
      return Real(0.0);
    return (r * r + r) / (r * r + Real(1.0));
  case TypesHllcScheme::Superbee:
    return math::max(Real(0.0),
                     math::max(math::min(Real(1.0), Real(2.0) * r),
                               math::min(Real(2.0), r)));
  }
  return Real(0.0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest::HllcScheme

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
