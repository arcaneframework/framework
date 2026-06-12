// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef MICROHYDROMODULE_H
#define MICROHYDROMODULE_H

#include "MicroHydro_axl.h"

using namespace Arcane;

/**
 * Represents a highly simplified Lagrangian hydrodynamics module:
 *   - the only supported cell type is the hexahedron,
 *   - no pseudo-viscosity supported,
 *   - the only supported characteristic length calculation type is the one using medians,
 *   - the only supported boundary condition type is imposing a velocity component on a surface,
 *   - the nodal mass is assumed constant and is not recalculated at each iteration,
 *   - no value consistency test (positive pressure, positive volume, ...) is performed.
 *  
 * The list of operations performed by the module is as follows:
 *   - calculation of pressure forces,
 *   - calculation of momentum,
 *   - taking into account boundary conditions,
 *   - node displacement,
 *   - calculation of new geometric values: cell volume, cell characteristic length, resultant forces at the vertices of each cell,
 *   - calculation of density,
 *   - calculation of pressure and energy using the equation of state. This calculation is performed by an ARCANE service.
 *     Two implementations are available for the service: perfect gas, and "stiffened" gas.
 *   - calculation of the new time step.
 * 
 */
class MicroHydroModule
: public ArcaneMicroHydroObject
{
 public:
  /** Class constructor */
  MicroHydroModule(const ModuleBuildInfo& mbi)
    : ArcaneMicroHydroObject(mbi) {}
  /** Class destructor */
  ~MicroHydroModule() {}
  
 public:
  /** 
   *  Initializes the module. 
   *  The initialization consists of two distinct parts:
   *  - the first part where the size of the array variables must be specified.
   *    In our case, these are \c m_cell_cqs and
   *    \c m_viscosity_force, both of which are cell variables possessing a
   *    value for every node of every cell. Since we only support hexahedrons, there
   *    are 8 values per cell,
   *  - the second part which consists of initializing the variables with
   *    their starting value. For the variables \c Pressure, \c Density, and
   *    \c AdiabaticCst, ARCANE initializes them directly
   *    from the dataset. The \c NodeCoord variable is also
   *    initialized by the architecture when reading the mesh. The
   *    other variables are calculated as follows:
   *    - the initial time step is given by the dataset,
   *    - the geometric values (characteristic length, volume, and
   *      resultant forces at the vertices) are calculated from the node coordinates,
   *    - the cell mass is calculated from its density andvolume,
   *    - the cell mass and nodal mass. A cell's mass is calculated from its density and volume,
   *    - the nodal mass is calculated by adding the contributions of
   *      each cell connected to a given node. Each cell
   *      contributes 1/8th of its mass to the nodal mass of each of its
   *      vertices,
   *    - the internal energy and the speed of sound are calculated based on the equation of state.
   */
  virtual void hydroStartInit();
  
  /** 
   * Calculates the contribution of pressure forces per node at the current
   * time \f$t^{n}\f$. For each node of each cell,
   * this is the pressure multiplied by the resultant force at that node.
   * Calculates the pressure forces at the current time \f$t^{n}\f$.
   */
  virtual void computePressureForce();
		
  /**
   * Calculates the force (\c m_force) applied to the nodes by adding the
   * possible contribution of pseudo-viscosity. Then calculates the new
   * velocity (\c m_velocity) at the nodes.
   */
  virtual void computeVelocity();
		
  /**
   * Applies the boundary conditions.
   * Boundary conditions depend on the dataset options. In this implementation,
   * a boundary condition has the following properties:
   * - a type: three types are supported: constraining the \f$x\f$ component
   *   of the velocity vector, constraining the \f$y\f$ component of the velocity
   *   vector, or constraining the \f$z\f$ component of the velocity vector,
   * - a value: this is a real number indicating the value of the constraint,
   * - a surface: this is the surface on which the constraint applies.
   * 
   * Applying boundary conditions therefore consists of fixing a component of a
   * velocity vector for every node of every face of every surface on which a
   * boundary condition is imposed.
   */		
  virtual void applyBoundaryCondition();
		
  /**
   * Modifies the coordinates (\c m_node_coord) of the nodes based on the
   * velocity vector and the time step.
   */
  virtual void moveNodes();
		
  /**
   * This entry point groups all the geometric calculations useful for the
   * scheme. In our case, this involves for each cell:
   * - calculating its characteristic length,
   * - calculating the resultant forces at its vertices,
   * - calculating its volume.
   
   * To optimize the calculation (cache usage), during each iteration on a cell,
   * the coordinates of its nodes and those of the center of its faces are stored locally.
   */
  virtual void computeGeometricValues();
  
  /**
   * Calculates the new value of the cell density, assuming that the mass
   * of a cell is constant over time. In this case, the new density is equal
   * to the mass divided by the new volume.
   */
  virtual void updateDensity();
		
  /**
   * This entry point calculates the internal energy, pressure, and speed of
   * sound within the cell by calling the equation of state service.
   */
  virtual void applyEquationOfState();
		
  /**
   * Determines the time step value for the next iteration. 
   * The time step is constrained by:
   * - the CFL value,
   * - the \c deltatMin() and \c deltatMax() values of the dataset,
   * - the final time value. During the last iteration, the time step must
   *   be such that we stop exactly at the time specified in the dataset
   *   (\c finalTime()).
   */
  virtual void computeDeltaT();

  /** Returns the version number of the module */
  virtual VersionInfo versionInfo() const { return VersionInfo(1,0,0); }
  
 private:
  /**
   * Calculates the resultant forces at the nodes of a hexahedral cell.
   * The method used is that of dividing into four triangles.
   * Method called by the \c computeGeometricValues() entry point
   */
  inline void computeCQs(Real3 node_coord[8],Real3 face_coord[6],const Cell& cell);
};

#endif
