// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SCALARBOUNDARYCONDITION_H
#define SCALARBOUNDARYCONDITION_H

#include "Numerics/BCs/IBoundaryCondition.h"

#include "Numerics/BCs/BoundaryConditionTypes.h"

#include "ExpressionParser/ExpressionDriver.h"

using namespace Arcane;

/*!
  \class ScalarBoundaryCondition
  \author Daniele Di Pietro <daniele-antonio.di-pietro@ifp.fr>
  \date 2007-07-30
  \brief Class for scalar boundary conditions.

  Under this name we collect the standard boundary conditions for diffusion 
  and advection-diffusion problems
*/

class ScalarBoundaryCondition : public IBoundaryCondition {
 public:
  struct Error {
    std::string msg;
    Error(const std::string& _msg) : msg(_msg) {}
  };
  
 public:
  //! Costructor from tag and value
  ScalarBoundaryCondition(BoundaryConditionTypes::eType type, 
                          const expression_parser::Expression& value) :
    m_type(type), m_value(value) {}
  //! Destructor
  virtual ~ScalarBoundaryCondition() {}

 public:
  //! Evaluate boundary condition
  Real eval(const Real3& x = Real3(0, 0, 0), Real t = 0) { return m_value.eval(x, t); }
  //! Return the type
  inline BoundaryConditionTypes::eType getType() { return m_type; }

 private:
  //! Type
  BoundaryConditionTypes::eType m_type;
  //! Value
  expression_parser::Expression m_value;
};

#endif
