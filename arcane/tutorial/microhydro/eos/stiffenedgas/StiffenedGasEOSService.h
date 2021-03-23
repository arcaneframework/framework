// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef STIFFENEDGASEOSSERVICE_H
#define STIFFENEDGASEOSSERVICE_H

#include "StiffenedGasEOS_axl.h"
#include "IEquationOfState.h"

using namespace Arcane;

/**
 * Représente le modèle d'équation d'état <em>Stiffened Gas</em>
 */
class StiffenedGasEOSService 
: public ArcaneStiffenedGasEOSObject
, public IEquationOfState
{
public:
  /** Constructeur de la classe */
  StiffenedGasEOSService(const ServiceBuildInfo & sbi)
    : ArcaneStiffenedGasEOSObject(sbi) {}
  
  /** Destructeur de la classe */
  virtual ~StiffenedGasEOSService() {};
  
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
