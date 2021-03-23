#ifndef IINTEGRATOR_H_
#define IINTEGRATOR_H_

namespace Arcane {}
using namespace Arcane;

class INumericalModel ;

class IIntegrator
{
public :
  virtual ~IIntegrator() {}
  virtual bool integrate(INumericalModel* model, Integer sequence) = 0 ;
};
#endif /*IINTEGRATOR_H_*/
