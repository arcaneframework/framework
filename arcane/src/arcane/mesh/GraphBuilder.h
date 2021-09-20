/*
 * GraphDofs.h
 *
 *  Created on: 15 fev 2021
 *      Author: delhom
 */

#ifndef ARCANE_SRC_ARCANE_MESH_GRAPHBUILDER_H_
#define ARCANE_SRC_ARCANE_MESH_GRAPHBUILDER_H_

#include "arcane/IGraph2.h"
#include "arcane/mesh/GraphDofs.h"
#include "arcane/mesh/ParticleFamily.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GraphBuilder {
public :
 static IGraph2* createGraph(IMesh* mesh, String const& particle_family_name=ParticleFamily::defaultFamilyName())
 {
   return new mesh::GraphDofs(mesh,particle_family_name);
 };

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif /* ARCANE_SRC_ARCANE_MESH_GRAPHBUILDER_H_ */

