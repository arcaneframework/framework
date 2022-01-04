// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef POISSONMODULE_H
#define POISSONMODULE_H

#include "TypesPoisson.h"
#include "Poisson_axl.h"

using namespace Arcane;

/**
 * Représente un module qui calcule une diffusion numérique de la chaleur
 * suivant l'équation de Poisson dans un parallélépipéde.
 */
class PoissonModule : public ArcanePoissonObject
{
public:
    /** Constructeur de la classe */
    PoissonModule(const ModuleBuildInfo & mbi) : ArcanePoissonObject(mbi) { }

    /** Destructeur de la classe */
    ~PoissonModule() { }

public:
    /**
     * Initialise les températures dans le maillage en fonction de la
     * température fournie dans le jeu de données. 
     * Cette méthode est un point d'entrée du module enregistré sous le nom
     * \c InitTemperatures.
     */
    virtual void initTemperatures();

    /**
     * Diffuse des températures dans le maillage. 
     * Cette méthode est un point d'entrée du module enregistré sous le nom
     * \c PropagateTemperatures.
     */
    virtual void propagateTemperatures();

    /** Retourne le numéro de version du module */
    virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }

private:

    /** Prend en compte des conditions aux limites. */
    void applyBoundaryConditions();
};

#endif
