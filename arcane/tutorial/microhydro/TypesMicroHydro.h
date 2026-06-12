#ifndef TYPESMICROHYDRO_H
#define TYPESMICROHYDRO_H

#include <arcane/ItemGroup.h>
#include "eos/IEquationOfState.h"

struct TypesMicroHydro
{
    enum eBoundaryCondition
    {
        VelocityX, //!< Fixed X Velocity
        VelocityY, //!< Fixed Y Velocity
        VelocityZ, //!< Fixed Z Velocity
        Unknown //!< Unknown type
    };
};

#endif
