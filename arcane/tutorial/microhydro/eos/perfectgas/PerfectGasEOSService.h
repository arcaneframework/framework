// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef PERFECTGASEOSSERVICE_H
#define PERFECTGASEOSSERVICE_H

#include "PerfectGasEOS_axl.h"
#include "IEquationOfState.h"

using namespace Arcane;

/**
 * Represents the <em>Perfect Gas</em> equation of state model
 */
class PerfectGasEOSService 
: public ArcanePerfectGasEOSObject
, public IEquationOfState
{
public:
  /** Class constructor */
  PerfectGasEOSService(const ServiceBuildInfo & sbi)
    : ArcanePerfectGasEOSObject(sbi) {}
  
  /** Class destructor */
  virtual ~PerfectGasEOSService() {};

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
