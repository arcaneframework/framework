#ifndef TYPESTIMESTEPMNG_H
#define TYPESTIMESTEPMNG_H

#include <arcane/ItemGroup.h>

struct TypesTimeStepMng
{
    enum eTimeStepMngType
    {
        Geometric, //!< Type inconnu
        Arithmetic
    };
};

#endif

