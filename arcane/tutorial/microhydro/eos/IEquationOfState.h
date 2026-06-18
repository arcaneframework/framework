// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IEQUATIONOFSTATE_H
#define IEQUATIONOFSTATE_H

#include <arcane/ItemTypes.h>

using namespace Arcane;

/**
 * Interface for the equation of state model calculation service.
 */
class IEquationOfState
{
public:
  /** Constructor for the class */
  IEquationOfState() {};
  /** Destructor for the class */
  virtual ~IEquationOfState() {};
  
public:
  /** 
   *  Initializes the equation of state for the given cell group
   *  and calculates the speed of sound and the internal energy.
   */
  virtual void initEOS(const CellGroup & group) = 0;
  /** 
   *  Applies the equation of state to the given cell group
   *  and calculates the speed of sound and the pressure.
   */
  virtual void applyEOS(const CellGroup & group) = 0;
};

#endif
