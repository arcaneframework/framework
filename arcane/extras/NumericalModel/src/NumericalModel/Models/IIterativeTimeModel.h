// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IITERATIVETIMEMODEL_H
#define IITERATIVETIMEMODEL_H
/* Author : gratienj at mon Aug 25 15:14:45 2008
 * Interface of Time evolutionary model
 */

#include <arcane/VariableTypedef.h>

using namespace Arcane;

class ITimeMng ;
class ITimeStepMng ;

class IIterativeTimeModel
{
public:
  virtual ~IIterativeTimeModel() { }
  virtual void startTimeStep() = 0 ;
  virtual void endTimeStep() = 0 ;
  virtual void setTimeMng(ITimeMng* time_mng) = 0 ;
  virtual ITimeMng* getTimeMng() = 0 ;
  virtual void setTimeStepMng(ITimeStepMng* time_step_mng) = 0 ;
  virtual ITimeStepMng* getTimeStepMng() = 0 ;
};

class IEvolutionaryTimeModel
{
public:
  virtual ~IEvolutionaryTimeModel() { }

  virtual void initMaxVarCriteria(VariableArrayReal& dx1,
                                  VariableArrayReal& dx2,
                                  VariableArrayReal& dx) const = 0 ;
  
  //! calcul des variations max
  virtual void computeMaxVar(VariableArrayReal& dx) = 0 ;
};

#endif
