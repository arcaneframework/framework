// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef STIFFENEDGASEOSSERVICE_H
#define STIFFENEDGASEOSSERVICE_H

#include "StiffenedGasEOS_axl.h"
#include "IEquationOfState.h"

using namespace Arcane;

/**
 * Represents the <em>Stiffened Gas</em> equation of state model.
 */
class StiffenedGasEOSService 
: public ArcaneStiffenedGasEOSObject
, public IEquationOfState
{
public:
  /** Constructor for the class */
  StiffenedGasEOSService(const ServiceBuildInfo & sbi)
    : ArcaneStiffenedGasEOSObject(sbi) {}
  
  /** Destructor for the class */
  virtual ~StiffenedGasEOSService() {};
  
public:
  /** 
   *  Initializes the equation of state for the cell group passed as an argument
   *  and calculates the speed of sound and internal energy. 
   */
  virtual void initEOS(const CellGroup & group);
  /** 
   *  Applies the equation of state to the cell group passed as an argument
   *  and calculates the speed of sound and pressure. 
   */
  virtual void applyEOS(const CellGroup & group);
};

#endif
