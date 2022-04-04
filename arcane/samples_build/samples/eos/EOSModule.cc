// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <arcane/ITimeLoopMng.h>

#include "IEquationOfState.h"
#include "EOS_axl.h"

using namespace Arcane;

namespace EOS
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Module pour tester les services d'EOS.
 */
class EOSModule
: public ArcaneEOSObject
{
 public:
  //! Constructeur
  EOSModule(const ModuleBuildInfo & mbi) 
  : ArcaneEOSObject(mbi) { }

  //! Destructeur
  ~EOSModule() = default;

 public:

  //! Point d'entrée d'initialsation des EOS.
  void initEOS() override;

  //! Point d'entrée de calcul des EOS.
  void computeEOS() override;

  //! Numéro de version du module
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EOSModule::
initEOS()
{
  // Initialise l'énergie et la vitesse du son
  IEquationOfState* x = options()->eosModel();
  x->initEOS(allCells(),m_pressure,m_adiabatic_cst,m_density,
             m_internal_energy,m_sound_speed);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EOSModule::
computeEOS()
{
  info() << "Compute EOS !";

  // Pour arrêter le calcul après 50 itérations
  if (m_global_iteration()>50)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  IEquationOfState* x = options()->eosModel();
  x->applyEOS(allCells(),m_adiabatic_cst,m_density,
              m_internal_energy,m_pressure,m_sound_speed);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_EOS(EOSModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}
