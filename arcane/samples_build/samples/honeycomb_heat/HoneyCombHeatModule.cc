// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include "HoneyCombHeat_axl.h"
#include <arcane/ITimeLoopMng.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module HoneyCombHeatModule.
 */
class HoneyCombHeatModule
: public ArcaneHoneyCombHeatObject
{
 public:

  explicit HoneyCombHeatModule(const ModuleBuildInfo& mbi)
  : ArcaneHoneyCombHeatObject(mbi)
  {}

 public:

  /*!
   * \brief Méthode appelée à chaque itération.
   */
  void compute() override;
  /*!
   * \brief Méthode appelée lors de l'initialisation.
   */
  void startInit() override;

  /** Retourne le numéro de version du module */
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 private:

  void _applyBoundaryCondition();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyCombHeatModule::
compute()
{
  info() << "Module HoneyCombHeatModule COMPUTE";

  // Stop code after 10 iterations
  if (m_global_iteration() > 500) {
    subDomain()->timeLoopMng()->stopComputeLoop(true);
    return;
  }

  // Mise a jour de la temperature aux noeuds en prenant la moyenne
  // valeurs aux mailles voisines
  ENUMERATE_NODE (inode, allNodes()) {
    Node node = *inode;
    Real sumt = 0;
    for (Cell cell : node.cells())
      sumt += m_cell_temperature[cell];
    m_node_temperature[inode] = sumt / node.nbCell();
  }
  m_node_temperature.synchronize();

  // Mise a jour de la temperature aux mailles en prenant la moyenne
  // des valeurs aux noeuds voisins
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    Real sumt = 0;
    for (Node node : cell.nodes())
      sumt += m_node_temperature[node];
    m_cell_temperature[icell] = sumt / cell.nbNode();
  }

  _applyBoundaryCondition();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyCombHeatModule::
startInit()
{
  info() << "Module HoneyCombHeatModule INIT";
  m_cell_temperature.fill(0.0);
  m_node_temperature.fill(0.0);

  // Initialise le pas de temps à une valeur fixe
  m_global_deltat = 1.0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HoneyCombHeatModule::
_applyBoundaryCondition()
{
  // Les 10 premières mailles ont une température fixe.
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    if (cell.uniqueId() < 10)
      m_cell_temperature[cell] = 10000.0;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_HONEYCOMBHEAT(HoneyCombHeatModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
