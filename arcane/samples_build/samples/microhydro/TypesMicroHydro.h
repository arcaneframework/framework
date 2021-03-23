#ifndef TYPESMICROHYDRO_H
#define TYPESMICROHYDRO_H

#include <arcane/ItemGroup.h>
#include "eos/IEquationOfState.h"

struct TypesMicroHydro
{
  enum eBoundaryCondition
  {
    VelocityX, //!< Vitesse X fixée
    VelocityY, //!< Vitesse Y fixée
    VelocityZ, //!< Vitesse Z fixée
    Unknown //!< Type inconnu
  };
};

#endif

