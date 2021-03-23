#ifndef IADVECTIONOPERATOR_H
#define IADVECTIONOPERATOR_H

#include "Numerics/DiscreteOperator/IDiscreteOperator.h"

class IAdvectionOperator : public IDiscreteOperator {
 public:
  //! Velocity flux type
  typedef VariableFaceReal VelocityFluxType;

 public:
  virtual ~IAdvectionOperator() {}

 public:
  //! Form the discrete operator
  virtual void formDiscreteOperator(const VelocityFluxType& v) = 0;
};

#endif
