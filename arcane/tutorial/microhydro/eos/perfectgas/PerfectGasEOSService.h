// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef PERFECTGASEOSSERVICE_H
#define PERFECTGASEOSSERVICE_H

#include "PerfectGasEOS_axl.h"
#include "IEquationOfState.h"

using namespace Arcane;

/**
 * Représente le modèle d'équation d'état <em>Gaz Parfait</em>
 */
class PerfectGasEOSService 
: public ArcanePerfectGasEOSObject
, public IEquationOfState
{
public:
  /** Constructeur de la classe */
  PerfectGasEOSService(const ServiceBuildInfo & sbi)
    : ArcanePerfectGasEOSObject(sbi) {}
  
  /** Destructeur de la classe */
  virtual ~PerfectGasEOSService() {};

public:
  /** 
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne. 
   */
  virtual void initEOS(const CellGroup & group);
  /** 
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression. 
   */
  virtual void applyEOS(const CellGroup & group);
};

#endif
