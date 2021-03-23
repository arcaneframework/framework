// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
// -*- C++ -*-
#ifndef IDIVKGRADDISCRETEOPERATOR_H
#define IDIVKGRADDISCRETEOPERATOR_H

#include "Numerics/DiscreteOperator/IDiscreteOperator.h"

class IDivKGradDiscreteOperator : public IDiscreteOperator {
public:
  //! Absolute permeability type
  typedef VariableCellReal3x3 AbsolutePermeabilityType;

public:
  virtual ~IDivKGradDiscreteOperator() {}
  virtual const Integer & status() const = 0;
public:
  //! Form the discrete operator associated with the scalar permeability \f$\kappa\f$
  virtual void formDiscreteOperator(const VariableCellReal& k) = 0;
  //! Form the discrete operator associated with the diagonal permeability \f$\kappa\f$
  virtual void formDiscreteOperator(const VariableCellReal3& k) = 0;
  //! Form the discrete operator associated with the permeability \f$\kappa\f$
  virtual void formDiscreteOperator(const VariableCellReal3x3& k) = 0 ;
};

#endif
