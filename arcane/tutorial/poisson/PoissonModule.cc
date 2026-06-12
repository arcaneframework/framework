// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "PoissonModule.h"

#include <arcane/MathUtils.h>
#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PoissonModule::initTemperatures()
{
    // so that the extractions do not overlap
    m_global_deltat = 1;

    // initialization of temperature on all cells
    // ...

    // application of temperature at the boundaries
    applyBoundaryConditions();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PoissonModule::propagateTemperatures()
{
    Real max_delta_cell_t = 0;

    // update of temperature on cells
    // ...
    {
        // calculation of the new temperature
        // ...
        // new_cell_t
        // ...

	// we observe the difference
        Real delta_cell_t = math::abs(new_cell_t - m_cell_temperature[icell]);
        max_delta_cell_t = math::max(max_delta_cell_t, delta_cell_t);

	// update of temperature
        m_cell_temperature[icell] = new_cell_t;
    }
	
    // update of temperature at nodes
    // ...

    // Given the calculation, synchronization of temperature on cells is unnecessary
    // synchronization of temperature at nodes and reduction of the difference
    // ...
	
    // application of boundary conditions
    applyBoundaryConditions();
	
    // test for stopping the time loop
    if (max_delta_cell_t < 0.2) subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PoissonModule::applyBoundaryConditions()
{
    // loop over boundary conditions
    int nb_boundary_condition = options()->boundaryCondition.size();
    for (int i = 0; i < nb_boundary_condition; ++i)
    {
        FaceGroup face_group = options()->boundaryCondition[i]->surface();
        Real temperature = options()->boundaryCondition[i]->value();
        TypesPoisson::eBoundaryCondition type = options()->boundaryCondition[i]->type();

        // loop over faces of the surface
        ENUMERATE_FACE(iface, face_group)
        {
            const Face & face = * iface;
            Integer nb_node = face.nbNode();

            // loop over nodes of the face
            for (NodeEnumerator inode(face.nodes()); inode(); ++inode)
            {
                switch (type)
                {
                    case TypesPoisson::Temperature:
                        m_node_temperature[inode] = temperature;
                        break;
                    case TypesPoisson::Unknown:
                        break;
                }
            }
        }
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_POISSON(PoissonModule);
