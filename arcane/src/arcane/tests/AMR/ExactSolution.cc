// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExactSolution.cc                                            (C) 2000-2010 */
/*                                                                           */
/* Service de solutions analytiques utilisees pour estimer l'erreur AMR.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// <h1>Example - Insteady exact solution in 3D Domain</h1>
//
// Dans ce cas, la solution analytique est:
// u(r,\theta) = r^{2/3} * \sin ( (2/3) * \theta),
// l'indicateur d'erreur choisi est le gradient de la solution.

//  Boucle adaptative:
// 1- calcul de la solution analytique
// 2- calcul de l'erreur grid and projection of the solution to the new grid
// 3- marquage des mailles pour adaptation en fonction de l'erreur estimée
// 4- Effectuer l'adaptation de maillage.
// 5- projection des variables (ici solution)
// Dans le fichier axl , le paramètre
// "max_adapt_iters" controle le monbre de boucle de adaptation,
// "max_level" controle le niveau max de raffinement, et
// "refine_percentage" / "coarsen_percentage" determine le nombre de mailles
// à raffiner/déraffiné dans chaque itération d'adaptation.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using namespace Arcane;

// Choisir d'utiliser ou non la solution singulière
bool singularity = true;



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Definir la solution exacte.
Real exactSolution(const Real3& p)
{
	const Integer dim= 3;
  const Real x = p.x;
  const Real y = (dim > 1) ? p[1] : 0.;

  if (singularity)
    {
      // u_exact = r^(2/3)*sin(2*theta/3).
      Real theta = atan2(y,x);

      // theta doit être entre 0 <= theta <= 2*pi
      if (theta < 0)
        theta += 2. * 3.14159265;

      // étendre la solution au cas 3D
      const Real z = (dim > 2) ? p.z : 0;

      return pow(x*x + y*y, 1./3.)*sin(2./3.*theta) + z;
    }
  else
    {
      // The exact solution to a nonsingular problem,
      // good for testing ideal convergence rates
      const Real z = (dim > 2) ? p.z : 0;

      return cos(x) * exp(y) * (1. - z);
    }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


// Gradient de la solution exacte
Real3 exactGradient(const Real3& p)

{
	const Integer dim= 3;
  //
  Real3 gradu;

  // xy
  const Real x = p.x;
  const Real y = dim > 1 ? p.y : 0.;

  if (singularity)
    {
      // Le gradient n'est pas défini au pt x=0.
      ARCANE_ASSERT ((x != 0.),("Gradient is not defined at point 0."));

      // constantes tmp
      const Real tt = 2./3.;
      const Real ot = 1./3.;

      // Rayon au carré
      const Real r2 = x*x + y*y;

      // u_exact = r**(2/3)*sin(2*theta/3).
      Real theta = atan2(y,x);

      // theta doit être entre  0 <= theta <= 2*pi
      if (theta < 0)
        theta += 2. * 3.14159265;

      // du/dx
      gradu.x = tt*x*pow(r2,-tt)*sin(tt*theta) - pow(r2,ot)*cos(tt*theta)*tt/(1.+y*y/x/x)*y/x/x;

      // du/dy
      if (dim > 1)
        gradu.y = tt*y*pow(r2,-tt)*sin(tt*theta) + pow(r2,ot)*cos(tt*theta)*tt/(1.+y*y/x/x)*1./x;

      if (dim > 2)
        gradu.z = 1.;
    }
  else
    {
      const Real z = (dim > 2) ? p.z : 0;

      gradu[0] = -sin(x) * exp(y) * (1. - z);
      if (dim > 1)
        gradu[1] = cos(x) * exp(y) * (1. - z);
      if (dim > 2)
        gradu[2] = -cos(x) * exp(y);
    }

  return gradu;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real exact3DSolution(const Real3& p)
{
  // coords xyz
  const Real x = p.x;
  const Real y = p.y;
  const Real z = p.z;

  // valeur de la solution analytique
  return 4096.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 exact3DGradient(const Real3& p)

{
  // vecteur gradient de la solution u.
  Real3 gradu;

  // coords xyz
  const Real x = p.x;
  const Real y = p.y;
  const Real z = p.z;

  gradu.x = 4096.*2.*(x-x*x)*(1.-2.*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z);
  gradu.y = 4096.*2.*(x-x*x)*(x-x*x)*(y-y*y)*(1.-2.*y)*(z-z*z)*(z-z*z);
  gradu.z = 4096.*2.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(1.-2.*z);

  return gradu;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Hessien de la solution exacte
Real3x3 exact3DHessian(const Real3& p)
{
  // dérivée seconde de u.
  Real3x3 hessu;

  // coords xyz
  const Real x = p.x;
  const Real y = p.y;
  const Real z = p.z;

  hessu.x.x = 4096.*(2.-12.*x+12.*x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z);
  hessu.x.y = 4096.*4.*(x-x*x)*(1.-2.*x)*(y-y*y)*(1.-2.*y)*(z-z*z)*(z-z*z);
  hessu.x.z = 4096.*4.*(x-x*x)*(1.-2.*x)*(y-y*y)*(y-y*y)*(z-z*z)*(1.-2.*z);
  hessu.y.y = 4096.*(x-x*x)*(x-x*x)*(2.-12.*y+12.*y*y)*(z-z*z)*(z-z*z);
  hessu.y.z = 4096.*4.*(x-x*x)*(x-x*x)*(y-y*y)*(1.-2.*y)*(z-z*z)*(1.-2.*z);
  hessu.z.z = 4096.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(2.-12.*z+12.*z*z);

  // le Hessien est par construction symetrique
  hessu.y.x = hessu.x.y;
  hessu.z.x = hessu.x.z;
  hessu.z.y = hessu.y.z;

  return hessu;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

