// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef POISSONMODULE_H
#define POISSONMODULE_H

#include "TypesPoisson.h"
#include "Poisson_axl.h"

using namespace Arcane;

/**
 * Represents a module that calculates a numerical heat diffusion
 * following Poisson's equation in a parallelepiped.
 */
class PoissonModule : public ArcanePoissonObject
{
public:
    /** Constructor of the class */
    PoissonModule(const ModuleBuildInfo & mbi) : ArcanePoissonObject(mbi) { }

    /** Destructor of the class */
    ~PoissonModule() { }

public:
    /**
     * Initializes the temperatures in the mesh based on the temperature
     * provided in the dataset.
     * This method is an entry point of the module registered under the
     * name \c InitTemperatures.
     */
    virtual void initTemperatures();

    /**
     * Diffuses temperatures in the mesh. 
     * This method is an entry point of the module registered under the
     * name \c PropagateTemperatures.
     */
    virtual void propagateTemperatures();

    /** Returns the version number of the module */
    virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }

private:

    /** Takes into account boundary conditions. */
    void applyBoundaryConditions();
};

#endif
