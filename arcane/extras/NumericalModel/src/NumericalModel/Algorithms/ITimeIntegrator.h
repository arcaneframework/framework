#ifndef ITIMEINTEGRATOR_H_
#define ITIMEINTEGRATOR_H_

#include <arcane/utils/ArcaneGlobal.h>

using namespace Arcane;

class INumericalModel ;

class ITimeIntegrator
{
public :
  typedef enum
  {
    NoError,
    BadModelTypeError,
    MinTimeStepReachedError,
    TimeStepError
  } eErrorType ;
  virtual ~ITimeIntegrator() {}
  virtual bool integrate(INumericalModel* model, Integer sequence) = 0 ;
  virtual eErrorType getError() const = 0 ;
};
#endif /*ITIMEINTEGRATOR_H_*/
