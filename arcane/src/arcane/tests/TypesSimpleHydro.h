// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TypesSimpleHydro.h                                          (C) 2000-2020 */
/*                                                                           */
/* Types du module d'hydrodynamique.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANETEST_TYPESSIMPLEHYDRO_H
#define ARCANETEST_TYPESSIMPLEHYDRO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define SIMPLE_HYDRO_BEGIN_NAMESPACE  namespace SimpleHydro {
#define SIMPLE_HYDRO_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SimpleHydro
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TypesSimpleHydro
{
 public:

  enum eViscosity
  {
    ViscosityNo,
    ViscosityCellScalar
  };

  enum eBoundaryCondition
  {
    VelocityX, //!< Vitesse X fixée
    VelocityY, //!< Vitesse Y fixée
    VelocityZ, //!< Vitesse Z fixée
    Unknown    //!< Type inconnu
  };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IBoundaryCondition
{
 public:
  virtual ~IBoundaryCondition() = default;
 public:
  virtual FaceGroup getSurface() =0;
  virtual Real getValue() =0;
  virtual TypesSimpleHydro::eBoundaryCondition getType() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseOptionsSimpleHydro;

class SimpleHydroModuleBase
{
 public:
  virtual ~SimpleHydroModuleBase() = default;
 public:
  Real getDeltatInit();
  TypesSimpleHydro::eViscosity getViscosity();
  Real getViscosityLinearCoef();
  Real getViscosityQuadraticCoef();
  ConstArrayView<IBoundaryCondition*> getBoundaryConditions();
  Real getCfl();
  Real getVariationSup();
  Real getVariationInf();
  Real getDensityGlobalRatio();
  Real getDeltatMax();
  Real getDeltatMin();
  Real getFinalTime();
  Integer getBackwardIteration();
  bool isCheckNumericalResult();
 protected:
  void _setHydroOptions(CaseOptionsSimpleHydro* o) { m_options = o; }
 private:
  CaseOptionsSimpleHydro* m_options = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISimpleHydroService
{
 public:

  virtual ~ISimpleHydroService() = default;

 public:

  virtual void hydroBuild() = 0;
  virtual void hydroStartInit() = 0;
  virtual void hydroInit() = 0;
  virtual void hydroExit() = 0;

  virtual void computeForces() = 0;
  virtual void computeVelocity() = 0;
  virtual void computeViscosityWork() = 0;
  virtual void applyBoundaryCondition() = 0;
  virtual void moveNodes() = 0;
  virtual void computeGeometricValues() = 0;
  virtual void updateDensity() = 0;
  virtual void applyEquationOfState() = 0;
  virtual void computeDeltaT() = 0;

  virtual void setModule(SimpleHydroModuleBase* module) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace SimpleHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

