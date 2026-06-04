// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExactSolution.cc                                            (C) 2000-2010 */
/*                                                                           */
/* Service of analytical solutions used to estimate AMR error.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// <h1>Example - Insteady exact solution in 3D Domain</h1>
//
// In this case, the analytical solution is:
// u(r,\theta) = r^{2/3} * \sin ( (2/3) * \theta),
// the chosen error indicator is the gradient of the solution.

//  Adaptive loop:
// 1- calculation of the analytical solution
// 2- calculation of the error grid and projection of the solution to the new grid
// 3- marking of meshes for adaptation based on the estimated error
// 4- Perform mesh adaptation.
// 5- projection of variables (here solution)
// In the axl file, the parameter
// "max_adapt_iters" controls the number of adaptation loops,
// "max_level" controls the maximum refinement level, and
// "refine_percentage" / "coarsen_percentage" determine the number of meshes
// to refine/coarsen in each adaptation iteration.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using namespace Arcane;

// Choose whether or not to use the singular solution
bool singularity = true;



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Define the exact solution.
Real exactSolution(const Real3& p)
{
	const Integer dim= 3;
  const Real x = p.x;
  const Real y = (dim > 1) ? p[1] : 0.;

  if (singularity)
    {
      // u_exact = r^(2/3)*sin(2*theta/3).
      Real theta = atan2(y,x);

      // theta must be between 0 <= theta <= 2*pi
      if (theta < 0)
        theta += 2. * 3.14159265;

      // extend the solution to the 3D case
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


// Gradient of the exact solution
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
      // The gradient is not defined at point x=0.
      ARCANE_ASSERT ((x != 0.),("Gradient is not defined at point 0."));

      // temporary constants
      const Real tt = 2./3.;
      const Real ot = 1./3.;

      // Radius squared
      const Real r2 = x*x + y*y;

      // u_exact = r**(2/3)*sin(2*theta/3).
      Real theta = atan2(y,x);

      // theta must be between  0 <= theta <= 2*pi
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
  // xyz coordinates
  const Real x = p.x;
  const Real y = p.y;
  const Real z = p.z;

  // value of the analytical solution
  return 4096.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 exact3DGradient(const Real3& p)

{
  // gradient vector of the solution u.
  Real3 gradu;

  // xyz coordinates
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

// Hessian of the exact solution
Real3x3 exact3DHessian(const Real3& p)
{
  // second derivative of u.
  Real3x3 hessu;

  // xyz coordinates
  const Real x = p.x;
  const Real y = p.y;
  const Real z = p.z;

  hessu.x.x = 4096.*(2.-12.*x+12.*x*x)*(y-y*y)*(y-y*y)*(z-z*z)*(z-z*z);
  hessu.x.y = 4096.*4.*(x-x*x)*(1.-2.*x)*(y-y*y)*(1.-2.*y)*(z-z*z)*(z-z*z);
  hessu.x.z = 4096.*4.*(x-x*x)*(1.-2.*x)*(y-y*y)*(y-y*y)*(z-z*z)*(1.-2.*z);
  hessu.y.y = 4096.*(x-x*x)*(x-x*x)*(2.-12.*y+12.*y*y)*(z-z*z)*(z-z*z);
  hessu.y.z = 4096.*4.*(x-x*x)*(x-x*x)*(y-y*y)*(1.-2.*y)*(z-z*z)*(1.-2.*z);
  hessu.z.z = 4096.*(x-x*x)*(x-x*x)*(y-y*y)*(y-y*y)*(2.-12.*z+12.*z*z);

  // the Hessian is symmetric by construction
  hessu.y.x = hessu.x.y;
  hessu.z.x = hessu.x.z;
  hessu.z.y = hessu.y.z;

  return hessu;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
