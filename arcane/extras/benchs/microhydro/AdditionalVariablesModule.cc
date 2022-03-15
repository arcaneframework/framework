// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AdditionalVariablesModule.cc                                (C) 2000-2022 */
/*                                                                           */
/* Gestion des variables additionnelles pour le bench 'MicroHydro'.          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ModuleFactory.h"

#include "AdditionalVariables_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace MicroHydro
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module AdditionalVariables.
 */
class AdditionalVariablesModule
: public ArcaneAdditionalVariablesObject
{
 public:

  //! Constructeur
  explicit AdditionalVariablesModule(const ModuleBuildInfo& mb);

 public:

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 public:

  void doInit() override;
  void doExit() override;
  void doOneIteration() override;

 private:

  UniqueArray<VariableCellReal*> m_cell_variables;
  VariableCellArrayReal m_cell_2d_variable;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AdditionalVariablesModule::
AdditionalVariablesModule(const ModuleBuildInfo& sbi)
: ArcaneAdditionalVariablesObject(sbi)
, m_cell_2d_variable(VariableBuildInfo(sbi.meshHandle(),"Cell3DVariable"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdditionalVariablesModule::
doInit()
{
  info() << "AdditionalVariablesModule: init()";

  MeshHandle mesh_handle = defaultMeshHandle();
  Integer nb_cell_variable = options()->nbAdditionalCellVariable();

  info() << "NbAdditionalCellVariable = " << nb_cell_variable;

  for (Integer i = 0; i < nb_cell_variable; ++i) {
    String var_name = String("AdditionalCellVariable") + String::fromNumber(i);
    auto* x = new VariableCellReal(VariableBuildInfo(mesh_handle, var_name));
    m_cell_variables.add(x);

    // Initialise la variable avec des valeurs quelconques.
    VariableCellReal& current_var(*x);
    ENUMERATE_CELL (icell, allCells()) {
      Real v = static_cast<Real>((i + 1) * 5 + (icell.itemLocalId() * 2));
      current_var[icell] = v;
    }
  }

  {
    Integer nb_value = options()->cellArrayVariableSize();
    if (nb_value > 0) {
      m_cell_2d_variable.resize(nb_value);
      m_cell_2d_variable.fill(2.0);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdditionalVariablesModule::
doExit()
{
  // Détruit les variables additionnelles
  for (VariableCellReal* x : m_cell_variables)
    delete x;
  m_cell_variables.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AdditionalVariablesModule::
doOneIteration()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(AdditionalVariablesModule, AdditionalVariables);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace MicroHydro

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
