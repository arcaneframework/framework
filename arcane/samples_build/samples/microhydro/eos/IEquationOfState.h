// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef IEQUATIONOFSTATE_H
#define IEQUATIONOFSTATE_H

#include <arcane/ItemTypes.h>

using namespace Arcane;

/**
 * Interface du service du modèle de calcul de l'équation d'état.
 */
class IEquationOfState
{
public:
  /** Constructeur de la classe */
  IEquationOfState() {};
  /** Destructeur de la classe */
  virtual ~IEquationOfState() {};
  
public:
  /** 
   *  Initialise l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et l'énergie interne. 
   */
  virtual void initEOS(const CellGroup & group) = 0;
  /** 
   *  Applique l'équation d'état au groupe de mailles passé en argument
   *  et calcule la vitesse du son et la pression. 
   */
  virtual void applyEOS(const CellGroup & group) = 0;
};

#endif
