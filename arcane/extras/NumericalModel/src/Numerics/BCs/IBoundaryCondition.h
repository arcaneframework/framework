// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IBOUNDARYCONDITION_H
#define IBOUNDARYCONDITION_H

#include <map>

#include <arcane/VariableTypes.h>
#include <arcane/VariableTypedef.h>

#include "Numerics/BCs/BoundaryConditionTypes.h"

/*!
  \class IBoundaryCondition
  \author Daniele Di Pietro <daniele-antonio.di-pietro@ifp.fr>
  \date 2007-07-26
  \brief Base class for boundary conditions
*/

using namespace Arcane;

class IBoundaryCondition {
 public:
  //! Weight type
  typedef std::pair<Real, Real> WeightType;
  
 public:
  virtual ~IBoundaryCondition() {}

public:
  //! Return boundary condition type
  virtual BoundaryConditionTypes::eType getType() = 0;

 public:
  //! Evaluate boundary condition
  virtual Real eval(const Real3& x = Real3(0, 0, 0), Real t = 0) = 0;

};

#endif
