#ifndef IXYZTFUNCTION_H
#define IXYZTFUNCTION_H

#include <arcane/VariableTypes.h>
#include <arcane/VariableTypedef.h>

using namespace Arcane;

class IXYZTFunction {
 public:
  virtual ~IXYZTFunction() {}
  virtual Real eval(const Real3& P, Real t = 0) = 0;
};

#endif
