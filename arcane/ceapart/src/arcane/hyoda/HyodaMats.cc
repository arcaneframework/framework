// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HyodaMats.cc                                                (C) 2000-2013 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef _HYODA_PLUGIN_MATERIALS_H_
#define _HYODA_PLUGIN_MATERIALS_H_

#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"

#include "arcane/AbstractService.h"
#include "arcane/FactoryService.h"

#include "arcane/IVariableMng.h"
#include "arcane/SharedVariable.h"
#include "arcane/CommonVariables.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MeshMaterialModifier.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/MeshEnvironmentBuildInfo.h"
#include "arcane/materials/CellToAllEnvCellConverter.h"
#include "arcane/materials/MatItemVector.h"

#include "arcane/hyoda/Hyoda.h"
#include "arcane/hyoda/HyodaArc.h"
#include "arcane/hyoda/HyodaMix.h"
#include "arcane/hyoda/HyodaIceT.h"
#include "arcane/hyoda/IHyodaPlugin.h"

//#include <GL/osmesa.h>
#include "GL/glu.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class HyodaMix;
using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HyodaMats: public AbstractService,
                 public IHyodaPlugin{
public:
  HyodaMats(const ServiceBuildInfo& sbi):
    AbstractService(sbi),
    m_sub_domain(sbi.subDomain()),
    m_defaultMesh(m_sub_domain->defaultMesh()),
    m_interface_normal(VariableBuildInfo(m_defaultMesh,
                                         "InterfaceNormal")),
    m_interface_distance(VariableBuildInfo(m_defaultMesh,
                                           "InterfaceDistance2",
                                           IVariable::PNoDump|IVariable::PNoRestore))
  {}

  // **************************************************************************
  // * drawGlobalCell
  // **************************************************************************
  int drawGlobalCell(Cell cell, Real min, Real max, Real val){
    Real3 rgb;
    debug()<<"\t[HyodaMats::drawGlobalCell] Cell #"<<cell.localId();
    glBegin(GL_POLYGON);
    hyoda()->meshIceT()->setColor(min,max,val,rgb);
    ENUMERATE_NODE(node, cell->nodes()){
      glColor3d(rgb[0], rgb[1], rgb[2]);
      glVertex2d(m_defaultMesh->nodesCoordinates()[node].x,
                 m_defaultMesh->nodesCoordinates()[node].y);
    }
    glEnd();
    return 0;
  }
  
  // **************************************************************************
  // * draw
  // **************************************************************************
  int draw(IVariable *variable, Real min, Real max)
  {
    ARCANE_UNUSED(variable);

    IMeshMaterialMng *material_mng=IMeshMaterialMng::getReference(m_defaultMesh);
    MaterialVariableCellReal mat_density(MaterialVariableBuildInfo(material_mng,"Density"));

    // Itération sur tous les milieux et tous les matériaux d'une maille.
    ENUMERATE_ALLENVCELL(iallenvcell, material_mng, m_defaultMesh->allCells()){
      AllEnvCell all_env_cell = *iallenvcell;
      Cell global_cell = all_env_cell.globalCell();
      hyodaMix()->setCellOrigin(global_cell);
      
      info()<<"[HyodaMats::draw] aeCell #"
            << global_cell.localId()
            << ", nbEnvironment=" << all_env_cell.nbEnvironment();
      // Nous sommes dans le cas du plugin materials qui doit avoir qu'un seul milieu par maille
      if (all_env_cell.nbEnvironment()!=1)
        info() << "[HyodaMats::draw] all_env_cell.nbEnvironment()!=1";
      
      ENUMERATE_CELL_ENVCELL(ienvcell,all_env_cell){
        info()<<"\t[HyodaMats::draw] eCell #"
              << (*ienvcell).globalCell().localId()
              << ", nbMaterial=" << (*ienvcell).nbMaterial();
        // Si la maille en cours n'a qu'un seul matériau, on l'affiche en global
        if ((*ienvcell).nbMaterial()==1) drawGlobalCell((*ienvcell).globalCell(),
                                                        min, max,
                                                        mat_density[(*ienvcell).globalCell()]);
        // Si la maille a 2 materiaux
        // et que la normale est non nulle
        // et qu'on a une distance non nulle
        if ((*ienvcell).nbMaterial()==2
            && m_interface_normal[global_cell].abs()!=0.
            && m_interface_distance[global_cell].size()==1
            && m_interface_distance[global_cell].at(0)!=0.){
          Int32 rtn=hyodaMix()->xCellPoints(global_cell,
                                            m_interface_normal[global_cell],
                                            m_interface_distance[global_cell].at(0),0);
          info()<<"\t[HyodaMats::draw] Cell #"
                << global_cell.localId()
                << ", rtn=" << rtn
                << ", mat0="<< mat_density[(*ienvcell).cell(0)]
                << ", mat1="<< mat_density[(*ienvcell).cell(1)];
          hyodaMix()->xCellDrawInterface((*ienvcell).globalCell(),/*order*/0); 
        }
        /*ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
          MatCell mc = *imatcell;
          Int32 idx = mc.materialId();
          //m_present_material[global_cell] = m_present_material[global_cell] | (1<<idx);
          debug()<<"\t\t[HyodaMats::draw] mCell #"
                << (*imatcell).globalCell().localId()
                << ", materialId=" << idx;
                }*/
      }
    }
    return 0;
  }
  
private:
  ISubDomain *m_sub_domain;
  IMesh* m_defaultMesh;    
  VariableCellReal3 m_interface_normal;
  VariableCellArrayReal m_interface_distance;
 };

 
ARCANE_REGISTER_SUB_DOMAIN_FACTORY(HyodaMats, IHyodaPlugin, HyodaMats);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // _HYODA_PLUGIN_MATERIALS_H_
